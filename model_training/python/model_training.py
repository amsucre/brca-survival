from pathlib import Path

import matplotlib
import seaborn as sns
from sklearn.metrics import confusion_matrix

from common_utils import BRCADataset, BRCAClassifier, load_data, train_and_evaluate_model_accuracy, train_and_evaluate_model_loss


# ===== MAIN FUNCTIONS =====
def train_best_model(parameters: dict, train_fpath: str, test_fpath: str, n_epoch: int, metric: str = 'accuracy', which_loss: str = 'default', 
                     get_c_index: bool = False, early_stop: bool = False, save_model_path: str = '', save_pred_path: str = '', save_pred_test_path: str = '', split_path: str = ''):
    # Prep data
    x_train, y_train, t_train, c_train, s_train = load_data(train_fpath)
    x_test, y_test, t_test, c_test, s_test = load_data(test_fpath)
    dataset_train = BRCADataset(x_train, y_train, t_train, c_train, s_train)
    dataset_test = BRCADataset(x_test, y_test, t_test, c_test, s_test)

    # Define model
    model = BRCAClassifier(x_train.shape[1], parameters)
    
    #  Define save settings
    save_model = save_model_path is not None and save_model_path != ''
    save_pred = save_pred_path is not None and save_pred_path != ''
    save_pred_test = save_pred_test_path is not None and save_pred_test_path != ''
    
    # Create save folders (if missing)
    if save_model:
        Path(save_model_path).parent.mkdir(parents=True, exist_ok=True)

    if save_pred:
        Path(save_pred_path).parent.mkdir(parents=True, exist_ok=True)
        
    if save_pred_test:
        Path(save_pred_test_path).parent.mkdir(parents=True, exist_ok=True)

    # Train and evaluate the model
    if metric == 'loss':
        outcome, y_pred, y_true, loss_per_epoch, c_index_dict = train_and_evaluate_model_loss(model, parameters, dataset_train, dataset_test, n_epoch, 
                                                                                              which_loss, final=True, get_c_index=get_c_index, early_stop=early_stop,
                                                                                              save_model=save_model, save_model_path=save_model_path, 
                                                                                              save_pred=save_pred, save_pred_path=save_pred_path,
                                                                                              save_pred_test=save_pred_test, save_pred_test_path=save_pred_test_path,
                                                                                              split_path=split_path)
        accuracy_per_epoch = dict() 
    else: 
        outcome, y_pred, y_true, loss_per_epoch, accuracy_per_epoch, c_index_dict = train_and_evaluate_model_accuracy(model, parameters, dataset_train, dataset_test, n_epoch, 
                                                                                                                      which_loss, final=True, get_c_index=get_c_index, early_stop=early_stop,
                                                                                                                      save_model=save_model, save_model_path=save_model_path, 
                                                                                                                      save_pred=save_pred, save_pred_path=save_pred_path,
                                                                                                                      save_pred_test=save_pred_test, save_pred_test_path=save_pred_test_path,
                                                                                                                      split_path=split_path)
        
    return outcome, y_pred, y_true, loss_per_epoch, c_index_dict, accuracy_per_epoch


def plot_cmatrix(y_true, y_pred, out_dir: str = None):
    if out_dir:
        matplotlib.use('agg')
    else:
        matplotlib.use('tkagg')
        
    import matplotlib.pyplot as plt # Import after choosing backend

    # Compute and Plot Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1, 2, 3], yticklabels=[0, 1, 2, 3])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(out_dir) / 'confussion_matrix.png')
    else:
        plt.show()
