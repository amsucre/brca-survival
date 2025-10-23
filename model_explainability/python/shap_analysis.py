import json
import sys
from datetime import datetime
from pathlib import Path 

import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt # Import after choosing backend


# ===== AUX CLASSES =====
# Define Neural Network Model with Dynamic Hidden Layers >> ONLY for SHAP, return softmax probabilities instead of raw logits
class BRCAClassifierShap(nn.Module):
    def __init__(self, input_size: int, params: dict):
        super(BRCAClassifierShap, self).__init__()
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
        # return self.model(x)
        return torch.softmax(self.model(x), dim=1)
  

# ===== AUX FUNCTIONS =====
def load_data(data_path: str) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list):
    df = pd.read_csv(data_path, sep='\t', header=0, index_col=0)
    
    x_cols = [i for i in df.columns.tolist() if i not in ['OUTCOME', 'MONTHS', 'IS_CENSORED']]
    x = torch.tensor(df.loc[:, x_cols].values, dtype=torch.float32)
    y = torch.tensor(df.loc[:, 'OUTCOME'].values, dtype=torch.long)
    t = torch.tensor(df.loc[:, 'MONTHS'].values, dtype=torch.float32) if 'MONTHS' in df.columns else None
    c = torch.tensor(df.loc[:, 'IS_CENSORED'].values, dtype=torch.long) if 'IS_CENSORED' in df.columns else None
    s = df.index.tolist()
    
    return x, y, t, c, s
  
  
def model_kernel_wrapper(model, x, device):
        # Convert NumPy input to PyTorch Tensor and move to model's device
        x_input_tensor = torch.tensor(x, dtype=torch.float32).to(device)
        
        # Pass through the model
        with torch.no_grad():
            probs = model(x_input_tensor)

        # Convert PyTorch probabilities back to NumPy array for KernelExplainer
        return probs.cpu().numpy()
  

# ===== MAIN FUNCTIONS =====
# Load a given model from file and extract/calculate the requested data
def load_model_and_get_shap(model_file: str, test_fpath: str, train_fpath: str, out_dir: Path, shap_outcome: list, shap_explainer: str = 'deep',
                            aux_in_file: bool = True, model_input_size: int = None, model_param_dict: dict = None,
                            ):
    print(f'Loading {model_file}...')
    print(f'Expected outcomes: {shap_outcome}')
    
    # Get data stored in the model file
    if aux_in_file:   # All we need is stored in the model file
        checkpoint = torch.load(model_file, weights_only=True)

        model_input_size = checkpoint['input_size']
        model_param_dict = checkpoint['params']
        model_state_dict = checkpoint['model_state_dict']
    else:   # Only the model is stored in the model file. The rest is given as inputs
        model_state_dict = torch.load(model_file, weights_only=True)
        
    # Set device (cuda si tenemos GPU. cpu otherwise)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    # Load model
    model = BRCAClassifierShap(model_input_size, model_param_dict)
    model.load_state_dict(model_state_dict)
    model.eval()
    model.to(device)  # si tenemos GPU, mover a cuda

    # Define background set (random representative subset of the training set)
    x_train, _, _, _, _ = load_data(train_fpath)
    n_background_samples = 100 
    if x_train.size(0) < n_background_samples:
        print(f'Warning: Training set has only {x_train.size(0)} samples. Using all for background.')
        bkg_index = torch.arange(x_train.size(0))
    else:
        bkg_index = torch.randperm(x_train.size(0))[:n_background_samples]
    
    bkg_set = x_train[bkg_index].to(device)
    
    # Define testing set (our original test set)
    x_test, _, _, _, _ = load_data(test_fpath)
    
    n_test_samples = None
    if n_test_samples: 
        if x_test.size(0) < n_test_samples:
            test_index = torch.arange(x_test.size(0))
        else:
            test_index = torch.randperm(x_test.size(0))[:n_test_samples]
        test_set = x_test[test_index].to(device)
    else:
        test_set = x_test.to(device)
    
    # Initiate SHAP explainer
    if shap_explainer == 'deep':
        explainer = shap.DeepExplainer(model, bkg_set)
    elif shap_explainer == 'kernel':
        bkg_set_np = x_train[bkg_index].cpu().numpy() # For Kernel, must be numpy
        test_set = test_set.cpu().numpy()  # For Kernel, must be numpy
        explainer = shap.KernelExplainer(lambda x: model_kernel_wrapper(model, x, device), bkg_set_np)
    else:
        print(f'{shap_explainer} is not a valid SHAP explainer')
        return
      
    # Compute SHAP values
    shap_values = explainer.shap_values(test_set, check_additivity=False)   # list of 4 arrays (one per Q category), each of shape [num_shap_samples, num_features]
    
    # Convert SHAP values and test data for plotting
    shap_values_np = [] # Initialize as an empty list
    n_classes = shap_values.shape[2] # third dimension
    for i in range(n_classes):
        shap_values_i = shap_values[:, :, i]
        shap_values_np.append(shap_values_i)
        
    if shap_explainer == 'deep':
        test_set_np = test_set.cpu().numpy()
    else:  # kernel > no change needed
        test_set_np = test_set
    
    # Prep data for plotting
    df_train = pd.read_csv(train_fpath, sep='\t', header=0, index_col=0, nrows=5)
    feature_names = [i for i in df_train.columns.tolist() if i not in ['OUTCOME', 'MONTHS', 'IS_CENSORED']]
    class_names = ['Q1', 'Q2', 'Q3', 'Q4']
    
    out_dir.mkdir(parents=True, exist_ok=True)

    # CREATE AND STORE SHAP PLOTS
    if 'bar' in shap_outcome:
        # Global Feature Importance (Bar Plot) for overall impact across all categories
        # Calculate mean absolute SHAP values across all categories for each feature
        overall_mean_abs_shap_values = np.mean(np.abs(np.array(shap_values_np)), axis=0)
        shap.summary_plot(overall_mean_abs_shap_values,
                          test_set_np,
                          feature_names=feature_names,
                          plot_type='bar',
                          show=False)
        plt.title('Overall Global Feature Importance (Mean SHAP across all categories)')
        plt.tight_layout()
        plt.savefig(out_dir / f'bar_overall_importance_{shap_explainer}.png', dpi=300, bbox_inches='tight')
        plt.close() 
        
        print('* Overall bar plot DONE')
        
    if 'stacked' in shap_outcome:
        # Global Feature Importance (Bar Plot) for impact across eaach category, as a stacked barplot
        shap.summary_plot(shap_values,
                          test_set_np,
                          feature_names=feature_names,
                          class_names=class_names,
                          plot_type='bar',
                          show=False)
        plt.title('Global Feature Importance (SHAP values per category)')
        plt.tight_layout()
        plt.savefig(out_dir / f'bar_stacked_importance_{shap_explainer}.png', dpi=300, bbox_inches='tight')
        plt.close() 

        print('* Stacked bar plot DONE')
    
    if 'bee' in shap_outcome:
        # Global Feature Importance (Beeswarm Plot) for each category (Quartile)
        i = 1
        for shap_values_q in shap_values_np:
            shap.summary_plot(shap_values_q,
                              test_set_np,
                              feature_names=feature_names,
                              plot_type='beeswarm',
                              show=False)
            plt.title(f'SHAP Beeswarm Plot for Q{i} Prediction')
            plt.tight_layout()
            plt.savefig(out_dir / f'beeswarm_q{i}_{shap_explainer}.png', dpi=300, bbox_inches='tight')
            plt.close() 
            
            i += 1

        print('* Beeswarm plots DONE')
            
    if 'force' in shap_outcome:
        # Individual Prediction Explanation (Force Plot for the first sample for each Quartile)
        i = 1
        for shap_values_q in shap_values_np:
            base_value = explainer.expected_value[i-1].item() if shap_explainer == 'deep' else np.mean(model_kernel_wrapper(model, bkg_set_np, device), axis=0)[i-1]  
            shap.force_plot(base_value, # base value for quartile i
                            shap_values_q[0,:],     # SHAP values for first sample, for quartile i
                            test_set_np[0,:],       # Original feature values for first sample
                            feature_names=feature_names,
                            matplotlib=True, # Set to True for static plot in scripts
                            show=False)
            plt.title(f'SHAP Force Plot for Sample 0 (Q{i})')
            plt.tight_layout()
            plt.savefig(out_dir / f'force_sample0_q{i}_{shap_explainer}.png', dpi=300, bbox_inches='tight')
            plt.close() 
            
            i += 1

        print('* Force plots DONE')
    
    if 'water' in shap_outcome:
        # Individual Prediction Explanation (Waterfall Plot for the first sample for each Quartile)
        i = 1
        for shap_values_q in shap_values_np:
            base_value = explainer.expected_value[i-1].item() if shap_explainer == 'deep' else np.mean(model_kernel_wrapper(model, bkg_set_np, device), axis=0)[i-1]  
            shap.plots.waterfall(shap.Explanation(values=shap_values_q[0,:], # SHAP values for first sample, Q-i
                                                  base_values=base_value, # base value for quartile i
                                                  data=test_set_np[0,:],       # Original feature values for first sample
                                                  feature_names=feature_names),
                                 show=False)
            plt.title(f'SHAP Waterfall  Plot for Sample 0 (Q{i})')
            plt.tight_layout()
            plt.savefig(out_dir / f'waterfall_sample0_q{i}_{shap_explainer}.png', dpi=300, bbox_inches='tight')
            plt.close() 
            
            i += 1

        print('* Waterfall plots DONE')
    
    if 'dependance' in shap_outcome:
        # Feature Dependence Plot for the most important feature from the bar plot
        overall_shap_values_per_sample = np.mean(np.abs(np.array(shap_values_np)), axis=0)  # same as in bar plot. Mean per sample for the 4 quartiles
        overall_shap_values_global = np.mean(overall_shap_values_per_sample, axis=0)  # mean per feature, combining values for all samples
        most_important_feature_idx = np.argmax(overall_shap_values_global)
        i = 1
        for shap_values_q in shap_values_np:
            shap.dependence_plot(most_important_feature_idx,
                                 shap_values_q, # SHAP values for Q-i
                                 test_set_np,
                                 feature_names=feature_names,
                                 show=False)
            plt.title(f'SHAP Dependence Plot for {feature_names[most_important_feature_idx]} (Q{i})')
            plt.tight_layout()
            plt.savefig(out_dir / f'dependance_{feature_names[most_important_feature_idx]}_q{i}_{shap_explainer}.png', dpi=300, bbox_inches='tight')
            plt.close()
      
            i += 1

        print('* Best feature dependance plots DONE')
            
    # STORE SHAP DATA
    out_dir_data = out_dir / 'shap_data'
    out_dir_data.mkdir(parents=True, exist_ok=True)
    
    # # Save raw SHAP values 
    stacked_shap_values = np.stack(shap_values_np, axis=0) # Shape: [num_classes, num_samples, num_features]
    np.save(out_dir_data / f'shap_values_{shap_explainer}.npy', stacked_shap_values)
    
    # # Save features subset used for SHAP analysis
    np.save(out_dir_data / f'explained_features_{shap_explainer}.npy', test_set_np)
    
    # # Save Baseline/Expected Value (depending on explainer)
    # # # Check explainer type to get base_value correctly
    if shap_explainer == 'deep':
        base_values = explainer.expected_value.tolist() # Convert tensor to list for JSON
    elif shap_explainer == 'kernel':
        # For KernelExplainer, calculate the expected values (one for each category)
        base_values = np.mean(model_kernel_wrapper(model, bkg_set_np, device), axis=0).tolist()
    else:
        base_values = 'N/A - Explainer type not recognized'

    with open(out_dir_data / f'base_values_{shap_explainer}.json', 'w') as f:
        json.dump(base_values, f)
        
    # # Save Predicted Output for Explained Instances
    if shap_explainer == 'deep':
        test_data_tensor = torch.tensor(test_set_np, dtype=torch.float32).to(device)
        with torch.no_grad(): # Use no_grad for inference to save memory and computation
            predicted_probabilities = model(test_data_tensor).cpu().numpy()
    elif shap_explainer == 'kernel':
        # If using KernelExplainer, use the model_wrapper_for_kernel which takes NumPy input
        predicted_probabilities = model_kernel_wrapper(model, test_set_np, device)
    else:
        predicted_probabilities = None

    if predicted_probabilities is not None:
        np.save(out_dir_data / f'predicted_probabilities_{shap_explainer}.npy', predicted_probabilities)

    # # Save Feature Names
    with open(out_dir_data / f'feature_names_{shap_explainer}.json', 'w') as f:
        json.dump(feature_names, f)

    # # Save Analysis Configuration/Metadata
    metadata = {
        'analysis_date': datetime.now().isoformat(),
        'shap_version': shap.__version__,
        'pytorch_version': torch.__version__,
        'python_version': sys.version,
        'model_input_size': model_input_size,
        'model_params': model_param_dict,
        'explainer_type': type(explainer).__name__,
        'num_background_samples': n_background_samples,
        'num_explained_samples': test_set_np.shape[0],
        'model_path': model_file,
        'train_data_path': train_fpath,
        'test_data_path': test_fpath,
        'device': str(device)
    }
    
    with open(out_dir_data / f'analysis_metadata_{shap_explainer}.json', 'w') as f:
        json.dump(metadata, f, indent=4) # Use indent for readability
        
    print('* Raw data and metadata files DONE')

    
    

    
     



    


    
      
    