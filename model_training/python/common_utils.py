import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sksurv.metrics import concordance_index_censored
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader, random_split, Subset


# ===== DEFINE AUX CLASSES =====
# Custom Dataset Class
class BRCADataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor | None, c: torch.Tensor | None, sample: list):
        self.x = x
        self.y = y
        self.t = t
        self.c = c   # 0 is uncensored (death) | 1 is censored (alive)
        self.sample = sample

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, str):
        return self.x[idx], self.y[idx], self.t[idx], self.c[idx], self.sample[idx]


# Define Neural Network Model with Dynamic Hidden Layers
class BRCAClassifier(nn.Module):
    def __init__(self, input_size: int, params: dict):
        super(BRCAClassifier, self).__init__()
        num_layers = params['num_layers']
        hidden_sizes = params['hidden_sizes'] if 'hidden_sizes' in params else [params[f"hidden_size_{i}"] for i in range(num_layers)]
        layers = []
        prev_size = input_size
        for i in range(num_layers):
            layers.append(nn.Linear(prev_size, hidden_sizes[i]))
            layers.append(nn.ReLU())
            prev_size = hidden_sizes[i]
        layers.append(nn.Linear(prev_size, 4))  # Output layer (4 classes)
        self.model = nn.Sequential(*layers)

    def forward(self, x) -> torch.Tensor:
        return self.model(x)
      

# Define loss function class based on PORPOISE
class NLLSurvLoss(nn.Module):
    """
    The negative log-likelihood loss function for the discrete time to event model (Zadeh and Schmid, 2020).
    Code borrowed from https://github.com/mahmoodlab/Patch-GCN/blob/master/utils/utils.py
    Parameters
    ----------
    alpha: float
    eps: float
        Numerical constant; lower bound to avoid taking logs of tiny numbers.
    reduction: str
        Do we sum or average the loss function over the batches. Must be one of ['mean', 'sum']
    """
    def __init__(self, alpha=0.0, eps=1e-7, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.reduction = reduction

    def __call__(self, h, y, t, c):
        """
        Parameters
        ----------
        h: (n_batches, n_classes)
            The neural network output discrete survival predictions such that hazards = sigmoid(h).
        y_c: (n_batches, 2) or (n_batches, 3)
            The true time bin label (first column) and censorship indicator (second column).
        """

        return nll_loss(h=h, y=y.unsqueeze(dim=1), c=c.unsqueeze(dim=1),
                        alpha=self.alpha, eps=self.eps,
                        reduction=self.reduction)


# ===== DEFINE AUX FUNCTIONS =====
# Load Data Function
def load_data(data_path: str) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list):
    df = pd.read_csv(data_path, sep='\t', header=0, index_col=0)
    
    x_cols = [i for i in df.columns.tolist() if i not in ['OUTCOME', 'MONTHS', 'IS_CENSORED']]
    x = torch.tensor(df.loc[:, x_cols].values, dtype=torch.float32)
    y = torch.tensor(df.loc[:, 'OUTCOME'].values, dtype=torch.long)
    t = torch.tensor(df.loc[:, 'MONTHS'].values, dtype=torch.float32) if 'MONTHS' in df.columns else None
    c = torch.tensor(df.loc[:, 'IS_CENSORED'].values, dtype=torch.long) if 'IS_CENSORED' in df.columns else None
    s = df.index.tolist()
    
    return x, y, t, c, s


# Split train/val 
def split_train_val(train_val_data: BRCADataset, val_ratio: float = 0.2, split_path: str | Path = None):
    if split_path and Path(split_path).exists():
        # Read input data + get train(val lists)
        split_df = pd.read_csv(split_path, sep=',', header=0, index_col=0)
        train_list = split_df['train'].tolist() 
        val_list = [x for x in split_df['val'].tolist() if not pd.isna(x)]
        
        # Create aux dict: sample name+index
        sample_to_idx = {name: i for i, name in enumerate(train_val_data.sample)}

        # Get train and val indexes
        train_idx = [sample_to_idx[name] for name in train_list]
        val_idx   = [sample_to_idx[name] for name in val_list]

        # Split train/val according to indexes
        train_data = Subset(train_val_data, train_idx)
        val_data   = Subset(train_val_data, val_idx)
        
    else:   # Perfom random split here
        total_size = len(train_val_data)
        val_size = int(val_ratio * total_size)
        train_size = total_size - val_size
        generator = torch.Generator().manual_seed(42)  # Always split the same sets

        train_data, val_data = random_split(train_val_data, [train_size, val_size], generator=generator)
    
    return train_data, val_data

     
# Compute Class Weights
def get_class_weights(y_train: torch.Tensor) -> torch.Tensor:
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train.numpy()), y=y_train.numpy())
    return torch.tensor(class_weights, dtype=torch.float32)

  
# Train model and obtain necessary outputs
def train_model(model: BRCAClassifier, data_loader, criterion, which_loss: str, optimizer):
    # Init aux variables
    total_loss = 0
    output_list = list()
    time_list = list() 
    censor_list = list() 
    
    # Train model
    model.train()
    
    # Calculate loss + Optimize across epochs    
    for xb, yb, tb, cb, sb in data_loader:
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb, tb, cb) if which_loss == 'porpoise' else criterion(outputs, yb)  # different based on chosen loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        # Save outputs in case we need them for c_index
        output_list.append(outputs)
        time_list.append(tb)
        censor_list.append(cb)
    
    # Calculate AVG loss
    avg_loss = total_loss / len(data_loader)
    
    return model, avg_loss, output_list, time_list, censor_list
  

# Eval model and obtain necessary outputs
def eval_model(model: BRCAClassifier, data_loader, criterion, which_loss: str, get_accuracy: bool = False, final: bool = False, get_prob: bool = False, get_class: bool = False):
    # Init aux variables
    total_loss = 0
    output_list = list()
    time_list = list()  
    censor_list = list() 
    probs_list = list()
    sample_list = list()
    class_list = list()
    
    # Init aux variables (only for accuracy)
    correct, total = 0, 0
    
    # Init aux variables (only for final testing)
    y_pred, y_true = [], []
    
    # Eval model
    model.eval()
    with torch.no_grad():
        for xb, yb, tb, cb, sb in data_loader:
            outputs = model(xb)
            loss_batch = criterion(outputs, yb, tb, cb) if which_loss == 'porpoise' else criterion(outputs, yb)
            total_loss += loss_batch.item()
            
            if get_accuracy or final or get_prob or get_class:
                probs = torch.softmax(outputs, dim=1)  # probabilites per class
                predicted = torch.argmax(probs, dim=1) # predicted class
                correct += (predicted == yb).sum().item()
                total += yb.size(0)
                
            if final:
                y_pred.extend(predicted.cpu().numpy())
                y_true.extend(yb.cpu().numpy())
              
            # Save outputs in case we need them for c_index
            output_list.append(outputs)
            time_list.append(tb)
            censor_list.append(cb)
            
            # Save sample list. May be useful in many cases
            sample_list.append(sb)
            
            # Save probabilities per category
            if get_prob:
                probs_list.append(probs)
                
            if get_class:
                class_list.append(predicted)
                
            
    # Calculate AVG loss
    avg_loss = total_loss / len(data_loader)
    
    # Calculate accuracy (if needed)
    if get_accuracy:
        accuracy = correct / total
    else:
        accuracy = None
    
    # Return varies based on input
    if final:
        return avg_loss, output_list, time_list, censor_list, accuracy, y_pred, y_true, probs_list, sample_list, class_list
    else:
        return model, avg_loss, output_list, time_list, censor_list, accuracy, probs_list, sample_list, class_list


# Common function to train a given model according to given parameters and using given data
def train_and_evaluate_model_accuracy(model: BRCAClassifier, parameters: dict, data_train_val: BRCADataset, data_test: BRCADataset | None,
                                      n_epoch: int, which_loss: str, final: bool = False, has_pruner: bool = False, trial: optuna.Trial = None, 
                                      timeout: bool = False, get_c_index: bool = False, early_stop: bool = False, 
                                      save_model: bool = False, save_model_path: str | Path = None, 
                                      save_pred: bool = False, save_pred_path: str | Path = None, 
                                      save_pred_test: bool = False, save_pred_test_path: str | Path = None, 
                                      split_path: str | Path = None):
    # Record time
    start_time = time.time()
    max_time = 1800
    
    # Split set in train/val + Load the datasets
    data_train, data_val = split_train_val(data_train_val, split_path=split_path)
    train_loader = DataLoader(data_train, batch_size=parameters['batch_size'], shuffle=True, num_workers=1) 
    val_loader = DataLoader(data_val, batch_size=parameters['batch_size'], shuffle=True, num_workers=1)
    
    # Define loss function
    if which_loss == 'default':
        data_train_y = torch.stack([data_train_val[idx][1] for idx in data_train.indices])   # luego de hacer el split, necesitamos obtener asi la Y
        class_weights = get_class_weights(data_train_y)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    elif which_loss == 'porpoise':
        alpha = parameters['alpha'] if 'alpha' in parameters else 0.0
        criterion = NLLSurvLoss(alpha=alpha)
    else:
        return None

    # Define optimizer
    if parameters['optimizer'] == 'SGD_M':
        optimizer = optim.SGD(model.parameters(), lr=parameters['lr'], momentum=0.9, nesterov=True)
    elif parameters['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=parameters['lr'])  # SGD sin momentum
    elif parameters['optimizer'] == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=parameters['lr'])
    else:  # ADAM: default
        optimizer = optim.Adam(model.parameters(), lr=parameters['lr'])

    # Init values for (possible) early stopping
    best_loss_val = (0, float('inf'))  # Start loss at INF (will always improve at first) >> Nos vale tambien para calculate c_index
    epoch_wait = int(n_epoch*0.1)      # Epochs we can wait before with no improvement
    epoch_counter = 0    # Epochs that havn't improved
    best_model_state = None   # To store best model
        
    # Init values for (possibly) calculating c-index
    best_ci_train = (0, 'N/A')
    best_ci_val = (0, 'N/A')
    
    # Init values to store loss and accuracy history
    loss_per_epoch = {'train':dict(), 'val':dict(), 'test': 0}
    accuracy_per_epoch = {'val':dict(), 'test': 0}
    best_accuracy_val = 0
    
    # Init aux values for reporting results
    n_print = 25 if final else 5

    # Train the model n times, according to the given epochs
    for epoch in range(n_epoch):
        # Train model
        model, avg_loss_train, output_list_train, time_list_train, censor_list_train = train_model(model, train_loader, criterion, which_loss, optimizer)
      
        # Evaluate the model on the validation set
        model, avg_loss_val, output_list_val, time_list_val, censor_list_val, accuracy_val, probs_list_val, sample_list_val, _ = eval_model(model, val_loader, criterion, which_loss, get_accuracy=True, get_prob=save_pred)
        
        # Store best accuracy (for optimization)
        best_accuracy_val = max(best_accuracy_val, accuracy_val)
        
        # Print results
        if epoch==0 or (epoch+1) % n_print == 0:
            if final:
                print(f"Epoch {epoch} | Train Loss: {avg_loss_train:.4f}, Val Loss: {avg_loss_val:.4f}, Val Accuracy: {accuracy_val:.4f}")
            else: 
                optuna.logging.get_logger('optuna').info(f"Trial {trial.number} | Epoch {epoch} | Train Loss: {avg_loss_train:.4f}, Val Loss: {avg_loss_val:.4f}, Val Accuracy: {accuracy_val:.4f}")
              
        
        # If early_stop, check if loss has improved or not. Stop if hasn't for a while (epoch_wait)
        if early_stop:
            if avg_loss_val < best_loss_val[1]:
                best_loss_val = (epoch, avg_loss_val)
                epoch_counter = 0
                best_model_state = model.state_dict()  # Store best model
            else:
                epoch_counter += 1
                if epoch_counter >= epoch_wait:
                    print(f"Epoch {epoch} | No improvement for {epoch_counter} epochs.")
                    print(">>> Early stopping triggered.")
                    break

        # If save_model required, save everytime the loss immproves
        if save_model:
            if avg_loss_val <= best_loss_val[1]:  
                best_loss_val = (epoch, avg_loss_val)
                best_model_state = model.state_dict()  # Store best model
                model_features = data_train_val.x.shape[1]
                model_params = parameters
                save_dict = {'model_state_dict': best_model_state,
                             'input_size': model_features,
                             'params': model_params,
                             'loss': which_loss}
                torch.save(save_dict, save_model_path)  
                
        # If save_pred required, save sample IDs + corresponding predictions in file
        if save_pred:
            if avg_loss_val <= best_loss_val[1]:  
                best_loss_val = (epoch, avg_loss_val)
                save_pred_in_file(probs_list_val, sample_list_val, save_pred_path)

        # If c-index required, calculate for the best epoch (overwritten everytime the loss improves)    
        if get_c_index:
            if avg_loss_val <= best_loss_val[1]: 
                best_ci_train = (epoch, compute_c_index(output_list_train, time_list_train, censor_list_train))
                best_ci_val = (epoch, compute_c_index(output_list_val, time_list_val, censor_list_val))
                
                best_loss_val = (epoch, avg_loss_val)
                
                if final:
                    print(f"Epoch {epoch} | Loss: {best_loss_val[1]:.4f}, C-Index: {best_ci_train[1]:.4f} (TR), {best_ci_val[1]:.4f} (VAL)")
                # else:
                #     optuna.logging.get_logger('optuna').info(f"Trial {trial.number} | Epoch {epoch} | Loss: {best_loss_val[1]:.4f}, C-Index: {best_ci_train[1]:.4f} (TR), {best_ci_val[1]:.4f} (VAL)")
                
        # Store AVG loss per epoch and print, if training "best" model
        if final:
            loss_per_epoch['train'][epoch] = avg_loss_train
            loss_per_epoch['val'][epoch] = avg_loss_val
            accuracy_per_epoch['val'][epoch] = accuracy_val

        # If optuna is pruning, check if the accuracy (VAL) seems to be improving enough or not
        if has_pruner:
            trial.report(accuracy_val, epoch)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        # If timeout threshold was reached, prune the trial
        if timeout and (time.time() - start_time > max_time):
            optuna.logging.get_logger('optuna').info(f'Timeout in trial in epoch #{epoch}!')
            raise optuna.exceptions.TrialPruned()
    
    # If we performed early stopping, reload the model that best performed
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # If training best model, perform final testing + Return all outputs || If optimizing, only return accuracy    
    if final:
        test_loader = DataLoader(data_test, batch_size=parameters['batch_size'], shuffle=True, num_workers=1)
        avg_loss_test, output_list_test, time_list_test, censor_list_test, accuracy_test, y_pred, y_true, probs_list_test, sample_list_test, _ = eval_model(model, test_loader, criterion, which_loss, get_accuracy=True, final=True, get_prob=save_pred_test)

        if get_c_index:
            c_index_test = compute_c_index(output_list_test, time_list_test, censor_list_test)
        else:
            c_index_test = 'N/A'
            
        # If save_pred required, save sample IDs + corresponding predictions in file
        if save_pred_test:
            save_pred_in_file(probs_list_test, sample_list_test, save_pred_test_path)
            
        loss_per_epoch['test'] = avg_loss_test
        accuracy_per_epoch['test'] = accuracy_test
        best_c_index = {'train': best_ci_train[1], 'val': best_ci_val[1], 'test': c_index_test}
        
        return accuracy_test, y_pred, y_true, loss_per_epoch, accuracy_per_epoch, best_c_index
    else:
        return best_accuracy_val


def train_and_evaluate_model_loss(model: BRCAClassifier, parameters: dict, data_train_val: BRCADataset, data_test: BRCADataset,
                                  n_epoch: int, which_loss: str, final: bool = False, has_pruner: bool = False, trial: optuna.Trial = None, 
                                  timeout: bool = False, get_c_index: bool = False, early_stop: bool = False, 
                                  save_model: bool = False, save_model_path: str | Path = None, 
                                  save_pred: bool = False, save_pred_path: str | Path = None, 
                                  save_pred_test: bool = False, save_pred_test_path: str | Path = None, 
                                  split_path: str | Path = None):
    # Record time
    start_time = time.time()
    max_time = 1800
    
    # Split set in train/val + Load the datasets
    data_train, data_val = split_train_val(data_train_val, split_path=split_path)
    train_loader = DataLoader(data_train, batch_size=parameters['batch_size'], shuffle=True, num_workers=1) 
    val_loader = DataLoader(data_val, batch_size=parameters['batch_size'], shuffle=True, num_workers=1)
    
    # Define loss function
    if which_loss == 'default':
        data_train_y = torch.stack([data_train_val[idx][1] for idx in data_train.indices])   # luego de hacer el split, necesitamos obtener asi la Y
        class_weights = get_class_weights(data_train_y)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    elif which_loss == 'porpoise':
        alpha = parameters['alpha'] if 'alpha' in parameters else 0.0
        criterion = NLLSurvLoss(alpha=alpha)
    else:
        return None
    
    # Define optimizer
    if parameters['optimizer'] == 'SGD_M':
        optimizer = optim.SGD(model.parameters(), lr=parameters['lr'], momentum=0.9, nesterov=True)
    elif parameters['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=parameters['lr'])  # SGD sin momentum
    elif parameters['optimizer'] == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=parameters['lr'])
    else:  # ADAM: default
        optimizer = optim.Adam(model.parameters(), lr=parameters['lr'])

    # Init values for (possible) early stopping
    best_loss_val = (0, float('inf'))  # Start loss at INF (will always improve at first) >> Nos vale tambien para calculate c_index
    epoch_wait = int(n_epoch*0.1)      # Epochs we can wait before with no improvement
    epoch_counter = 0    # Epochs that havn't improved
    best_model_state = None   # To store best model
        
    # Init values for (possibly) calculating c-index
    best_ci_train = (0, 'N/A')
    best_ci_val = (0, 'N/A')
    
    # Init values to store loss and accuracy history
    loss_per_epoch = {'train':dict(), 'val':dict(), 'test': 0}
    best_loss_save = float('inf')
    
    # Init aux values for reporting results
    n_print = 25 if final else 5

    # Train the model n times, according to the given epochs
    for epoch in range(n_epoch):
        # Train model
        model, avg_loss_train, output_list_train, time_list_train, censor_list_train = train_model(model, train_loader, criterion, which_loss, optimizer)
      
        # Evaluate the model on the validation set
        model, avg_loss_val, output_list_val, time_list_val, censor_list_val, _, probs_list_val, sample_list_val, _ = eval_model(model, val_loader, criterion, which_loss, get_prob=save_pred)
        
        # Store best accuracy (for optimization)
        best_loss_save = min(best_loss_save, avg_loss_val)
        
        # Print results
        if epoch==0 or (epoch+1) % n_print == 0:
            if final:
                print(f"Epoch {epoch} | Train Loss: {avg_loss_train:.4f}, Val Loss: {avg_loss_val:.4f}")
            else: 
                optuna.logging.get_logger('optuna').info(f"Trial {trial.number} | Epoch {epoch} | Train Loss: {avg_loss_train:.4f}, Val Loss: {avg_loss_val:.4f}")
              
        # If early_stop, check if loss has improved or not. Stop if hasn't for a while (epoch_wait)
        if early_stop:
            if avg_loss_val < best_loss_val[1]:
                best_loss_val = (epoch, avg_loss_val)
                epoch_counter = 0
                best_model_state = model.state_dict()  # Store best model
            else:
                epoch_counter += 1
                if epoch_counter >= epoch_wait:
                    print(f"Epoch {epoch} | No improvement for {epoch_counter} epochs.")
                    print(">>> Early stopping triggered.")
                    break
                      
        # If save_model required, save everytime the loss immproves
        if save_model:
            if avg_loss_val <= best_loss_val[1]:   
                best_loss_val = (epoch, avg_loss_val)
                best_model_state = model.state_dict()  # Store best model
                model_features = data_train_val.x.shape[1]
                model_params = parameters
                save_dict = {'model_state_dict': best_model_state,
                             'input_size': model_features,
                             'params': model_params,
                             'loss': which_loss}
                torch.save(save_dict, save_model_path)
        
        # If save_pred required, save sample IDs + corresponding predictions in file
        if save_pred:
            if avg_loss_val <= best_loss_val[1]:   
                best_loss_val = (epoch, avg_loss_val)
                save_pred_in_file(probs_list_val, sample_list_val, save_pred_path)
                      
        # If c-index required, calculate for the best epoch (overwritten everytime the loss improves)    
        if get_c_index:
            if avg_loss_val <= best_loss_val[1]:  
                best_ci_train = (epoch, compute_c_index(output_list_train, time_list_train, censor_list_train))
                best_ci_val = (epoch, compute_c_index(output_list_val, time_list_val, censor_list_val))
                
                best_loss_val = (epoch, avg_loss_val)
                
                if final:
                    print(f"Epoch {epoch} | Loss: {best_loss_val[1]:.4f}, C-Index: {best_ci_train[1]:.4f} (TR), {best_ci_val[1]:.4f} (VAL)")
                # else:
                #     optuna.logging.get_logger('optuna').info(f"Trial {trial.number} | Epoch {epoch} | Loss: {best_loss_val[1]:.4f}, C-Index: {best_ci_train[1]:.4f} (TR), {best_ci_val[1]:.4f} (VAL)")
                
        # Store AVG loss per epoch and print, if training "best" model    
        if final:
            loss_per_epoch['train'][epoch] = avg_loss_train
            loss_per_epoch['val'][epoch] = avg_loss_val

        # If optuna is pruning, check if the loss seems to be improving enough or not
        if has_pruner:
            trial.report(avg_loss_val, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
                
        # If timeout threshold was reached, prune the trial
        if timeout and (time.time() - start_time > max_time):
            optuna.logging.get_logger('optuna').info(f'Timeout in trial in epoch #{epoch}!')
            raise optuna.exceptions.TrialPruned()
          
    # If we performed early stopping, reload the model that best performed
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # If training best model, perform final testing + Return all outputs || If optimizing, only return loss (VAL)    
    if final:
        test_loader = DataLoader(data_test, batch_size=parameters['batch_size'], shuffle=True, num_workers=1)
        avg_loss_test, output_list_test, time_list_test, censor_list_test, _, y_pred, y_true, probs_list_test, sample_list_test, _ = eval_model(model, test_loader, criterion, which_loss, final=True, get_prob=save_pred_test)

        if get_c_index:
            c_index_test = compute_c_index(output_list_test, time_list_test, censor_list_test)
        else:
            c_index_test = 'N/A'
            
        # If save_pred required, save sample IDs + corresponding predictions in file
        if save_pred_test:
            save_pred_in_file(probs_list_test, sample_list_test, save_pred_test_path)
            
        loss_per_epoch['test'] = avg_loss_test
        best_c_index = {'train': best_ci_train[1], 'val': best_ci_val[1], 'test': c_index_test}
        
        return avg_loss_test, y_pred, y_true, loss_per_epoch, best_c_index
    else:
        return best_loss_save


# PORPOISE: Loss auxiliary function
def nll_loss(h, y, c, alpha=0.0, eps=1e-7, reduction='mean'):
    """
    The negative log-likelihood loss function for the discrete time to event model (Zadeh and Schmid, 2020).
    Code borrowed from https://github.com/mahmoodlab/Patch-GCN/blob/master/utils/utils.py
    Parameters
    ----------
    h: (n_batches, n_classes)
        The neural network output discrete survival predictions such that hazards = sigmoid(h).
    y: (n_batches, 1)
        The true time bin index label.
    c: (n_batches, 1)
        The censoring status indicator.
    alpha: float
        This parameter helps balancing loss for censored/uncensored. Decides which term is more important. [0: Treats censored and uncensored equally, 1: Only considers uncensored, 0.5: 1/2 weight for combined & 1/2 for uncensored]
    eps: float
        Numerical constant; lower bound to avoid taking logs of tiny numbers.
    reduction: str
        Do we sum or average the loss function over the batches. Must be one of ['mean', 'sum']
    References
    ----------
    Zadeh, S.G. and Schmid, M., 2020. Bias in cross-entropy-based training of deep survival networks. IEEE transactions on pattern analysis and machine intelligence.
    """
    # make sure these are LongTensors (int)
    y = y.long()
    c = c.long()
    
    # Compute hazards
    hazards = torch.sigmoid(h)

    # Compute cumulative survival function
    S = torch.cumprod(torch.clamp(1 - hazards, min=eps), dim=1)  # Use torch clamp to avoid having  that may rise error

    # Append 1 at the begining (why?)
    S_padded = torch.cat([torch.ones_like(c), S], 1)
    # # S(-1) = 0, all patients are alive from (-inf, 0) by definition
    # # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    # # hazards[y] = hazards(1)
    # # S[1] = S(1)

    # Extract survival probabilities
    s_prev = torch.gather(S_padded, dim=1, index=y).clamp(min=eps)
    h_this = torch.gather(hazards, dim=1, index=y).clamp(min=eps)
    s_this = torch.gather(S_padded, dim=1, index=y+1).clamp(min=eps)

    # Compute loss for censored and uncensored data
    uncensored_loss = -(1 - c) * (torch.log(s_prev) + torch.log(h_this))
    censored_loss = - c * torch.log(s_this)
    
    # Combine losses, based on alpha value
    neg_l = censored_loss + uncensored_loss
    if alpha is None:
        loss = neg_l
    else: 
        loss = (1 - alpha) * neg_l + alpha * uncensored_loss
        # alpha = 0 >> Considers censored and uncensored loss equally
        # alpha = 1 >> Only considers uncensored loss
        # 0 < alpha < 1 >> Considers both terms (merged + only uncensored)
        
    # Compute loss reduction (Combine)
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    else:
        raise ValueError("Bad input for reduction: {}".format(reduction))

    return loss
  

# Compute risk array for c-index calculations  
def compute_risk(logit_list: list[torch.Tensor]) -> np.array:
    logit_tensor = torch.cat(logit_list, dim=0)   # We merge a list of tensors into a single one
    hazard_tensor = torch.sigmoid(logit_tensor)  # sigmoid returns probabilities of death on each period. Values do not sum 1 (different to softmax)
    survival_tensor = torch.cumprod(1 - hazard_tensor, dim=1)  # probability of survival (1-death) after each time period. Multiplies current x previous
    risk_array = -torch.sum(survival_tensor, dim=1).detach().cpu().numpy() # cumulative risk (inverse of survival sum) > More negative, less risk. So when we sort by risk, those on top have more risk of dying earlier

    return risk_array
 

# Compute concordance index
def compute_c_index(logit_list: list[torch.Tensor], time_list: list[torch.Tensor], censor_list: list[torch.Tensor]) -> float:
    # Calculate risks array
    risk_array = compute_risk(logit_list)
    
    # Convert tensors to arrays
    time_array = torch.cat(time_list, dim=0).detach().cpu().numpy()
    censor_array = torch.cat(censor_list, dim=0).detach().cpu().numpy()
    
    # Obtain event boolean (True=died) from censorship array
    event_bool_array = (1-censor_array).astype(bool)
    
    # Calculate C-Index
    # The function returns c_index, n_concordant, n_discordant, n_tied_risk, n_tied_time. We keep only first
    c_index = concordance_index_censored(event_bool_array, time_array, risk_array)[0]
    
    return c_index


# Save calculated predictions in a given file
def save_pred_in_file(pred_list_nested: list[list], sample_list_nested: list[list], fpath: str | Path, multi_pred: bool = True, pred_in_array: bool = False):
    if multi_pred: # 4 preds per sample, in a nested list
        pred_list = [', '.join(map(str, x.tolist())) for sublist in pred_list_nested for x in sublist]
    elif pred_in_array:  # 1 pred per sample, in a 1D numpy array
        pred_list = pred_list_nested.tolist()
    else:   # 1 pred per sample, in a nested list
        pred_list = [x for sublist in pred_list_nested for x in sublist]
        
    sample_list = [x for sublist in sample_list_nested for x in sublist]
    
    # Make sure outdir exists
    fpath = Path(fpath)
    fpath.parent.mkdir(parents=True, exist_ok=True)
    
    # Save data
    df = pd.DataFrame({'SAMPLE': sample_list, 'PREDICTION': pred_list})
    df.to_csv(fpath, sep='\t', index=False, header=True)


# Load a given model from file and extract/calculate the requested data
def load_model_and_get_outcomes(model_file: str, outcome_dict: dict, 
                                aux_in_file: bool = True, model_input_size: int = None, model_param_dict: dict = None, model_loss: str = None,
                                test_fpath: str = None, train_val_fpath: str = None, split_fpath: str = None):
    print(f'Loading {model_file}...')
    print(f'Expected outcomes: {outcome_dict.keys()}')
    
    # Get data stored in the model file
    if aux_in_file:   # All we need is stored in the model file
        checkpoint = torch.load(model_file, weights_only=True)

        model_input_size = checkpoint['input_size']
        model_param_dict = checkpoint['params']
        model_loss = checkpoint['loss']
        model_state_dict = checkpoint['model_state_dict']
    else:   # Only the model is stored in the model file. The rest is given as inputs
        model_state_dict = torch.load(model_file, weights_only=True)

    # Load model
    model = BRCAClassifier(model_input_size, model_param_dict)
    model.load_state_dict(model_state_dict)
    model.eval()
    
    # Define lists of outcomes to obtain for each dataset
    # output_dict has param_name:output_fpath
    outcome_val = [k for k in outcome_dict.keys() if 'val' in k]
    outcome_test = [k for k in outcome_dict.keys() if 'test' in k]
    
    # Get all data that was requested and store in given file
    if train_val_fpath:
        model, _, output_list_val, _, _, _, probs_list_val, sample_list_val, class_list_val, val_loader = load_data_and_eval_model(model, 'val', train_val_fpath, model_param_dict, model_loss, outcome_val, split_fpath)
        
        if 'val_prob_4q' in outcome_val:
            save_pred_in_file(probs_list_val, sample_list_val, outcome_dict['val_prob_4q'])
            
        if 'val_pred_class' in outcome_val:
            save_pred_in_file(class_list_val, sample_list_val, outcome_dict['val_pred_class'], multi_pred=False)
            
        if 'val_risk' in outcome_val:
            risk_array = compute_risk(output_list_val)
            save_pred_in_file(risk_array, sample_list_val, outcome_dict['val_risk'], multi_pred=False, pred_in_array=True)
            
        if 'val_final_hidden' in outcome_val:
            activation_df = get_penultimate_activations(model, val_loader)
            activation_df.to_csv(outcome_dict['val_final_hidden'], sep='\t')
            
        if 'val_4q_raw' in outcome_val:
            save_pred_in_file(output_list_val, sample_list_val, outcome_dict['val_4q_raw'])

     
    if test_fpath:
        model, _, output_list_test, _, _, _, probs_list_test, sample_list_test, class_list_test, test_loader = load_data_and_eval_model(model, 'test', test_fpath, model_param_dict, model_loss, outcome_test)
        
        if 'test_prob_4q' in outcome_test:
            save_pred_in_file(probs_list_test, sample_list_test, outcome_dict['test_prob_4q'])
            
        if 'test_pred_class' in outcome_test:
            save_pred_in_file(class_list_test, sample_list_test, outcome_dict['test_pred_class'], multi_pred=False)
            
        if 'test_risk' in outcome_test:
            risk_array = compute_risk(output_list_test)
            save_pred_in_file(risk_array, sample_list_test, outcome_dict['test_risk'], multi_pred=False, pred_in_array=True)
            
        if 'test_final_hidden' in outcome_test:
            activation_df = get_penultimate_activations(model, test_loader)
            activation_df.to_csv(outcome_dict['test_final_hidden'], sep='\t') 
            
        if 'test_4q_raw' in outcome_test:
            save_pred_in_file(output_list_test, sample_list_test, outcome_dict['test_4q_raw'])


# Aux function to load given dataset and eval model
def load_data_and_eval_model(model, which_set, data_fpath, param_dict, which_loss, outcome_dict, split_path: str = None):  
    # Prep and load dataset
    x, y, t, c, s = load_data(data_fpath)
    dataset = BRCADataset(x, y, t, c, s)
    if which_set == 'val':
        _, dataset = split_train_val(dataset, split_path=split_path)
        
    data_loader = DataLoader(dataset, batch_size=param_dict['batch_size'], shuffle=True, num_workers=1)
    
    # Define loss function
    if which_loss == 'default':
        class_weights = get_class_weights()
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:  #  'porpoise'
        alpha = param_dict['alpha'] if 'alpha' in param_dict else 0.0
        criterion = NLLSurvLoss(alpha=alpha)
        
    # Define data to get
    get_prob = f'{which_set}_prob_4q' in outcome_dict
    get_class = f'{which_set}_pred_class' in outcome_dict
    
    # Eval model + Get needed data
    return (*eval_model(model, data_loader, criterion, which_loss, get_prob=get_prob, get_class=get_class), data_loader)
  

# Get activations from the last hidden layer and return as a DF
def get_penultimate_activations(model, data_loader):
    # Init vars
    model.eval()
    activations_list = []
    sample_names = []

    # Get all model layers except the last one (multiclassifier)
    intermediate_model = nn.Sequential(*list(model.model.children())[:-1])

    with torch.no_grad():
        for xb, _, _, _, sb in data_loader:  # ignore y, t, c
            features = intermediate_model(xb)
            activations_list.append(features.cpu())
            sample_names.extend(sb)

    # Merge all results in a single tensor
    activations_tensor = torch.cat(activations_list, dim=0)
    
    # Store data in a pandas DF, having sample names as index
    df = pd.DataFrame(activations_tensor.numpy(), index=sample_names)
    df.index.name = 'SAMPLE'
    
    return df
  