from __future__ import print_function
import numpy as np

from pathlib import Path
import argparse
import os

import pandas as pd


### Internal Imports
from PORPOISE.datasets.dataset_survival import Generic_MIL_Survival_Dataset
from PORPOISE.models.model_porpoise import PorpoiseMMF, PorpoiseAMIL
from PORPOISE.models.model_early import EarlyMMF
from PORPOISE.models.model_genomic import SNN
from PORPOISE.utils.utils import get_evaluation_loader
from MGCT_MCAT.utils.utils import get_evaluation_loader as get_evaluation_loader_MGCT
from MGCT_MCAT.models.MGCT import MGCT
from MGCT_MCAT.models.mm_baselines import MCAT_Surv
from MGCT_MCAT.models.new_MGCT import MGCT_Surv
from MGCT_MCAT.datasets.dataset_survival import Generic_MIL_Survival_Dataset as Generic_MIL_Survival_Dataset_MGCT
from utils_benchmark import summary_survival
from config import EvaluateConfig
from config import InferenceConfig

### PyTorch Imports
import torch


def get_survival_results(set_dataloader, model_dict, model_type, fold_num):
    if model_type == 'porpoise_mmf':
        model = PorpoiseMMF(**model_dict)
    elif model_type == 'early_mmf':
        model = EarlyMMF(**model_dict)
    elif model_type == 'porpoise_amil':
        model = PorpoiseAMIL(**model_dict)
    elif model_type == 'snn':
        model = SNN(**model_dict)
    elif model_type == 'mgct':
        model = MGCT(**model_dict)
    elif model_type == 'new_mgct':
        model = MGCT_Surv(**model_dict)
    elif model_type == 'mcat':
        model = MCAT_Surv(**model_dict)
        
    model = model.to(torch.device('cuda'))
    print('results dir here', args.results_dir)
    print('results evalu here', args.results_dir_evaluation)

    state_dict = torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(fold_num)))

    # Remove "module." prefix from keys as not using data parallel when loading the model
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(new_state_dict)

    # model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(fold_num))))
    
    results_dict, c_index = summary_survival(model, set_dataloader, fold_num, model_type, str(Path(InferenceConfig.save_csv_path, 'patches_TCGA_2048', 'amil_features', 'uni', str(fold_num))))
    
    return results_dict, c_index
    

def main(args):
    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)
            
    latest_train_cindex = []
    latest_val_cindex = []
    latest_test_cindex = []

    folds = np.arange(5)

    for i in folds:
        ## Gets the Train + Val Dataset Loader.
        train_dataset, val_dataset, test_dataset = dataset.return_all_splits(from_id=False, csv_path='{}/splits_{}.csv'.format(args.split_dir, i))
        train_dataset.set_split_id(split_id=i)
        val_dataset.set_split_id(split_id=i)
        
        #pdb.set_trace()
        print('training: {}, validation: {}'.format(len(train_dataset), len(val_dataset)))

        if args.model_type=='porpoise_mmf' or args.model_type=='porpoise_amil' or args.model_type=='snn' or args.model_type=='early_mmf':
            train_loader = get_evaluation_loader(train_dataset)
            val_loader = get_evaluation_loader(val_dataset)
            test_loader = get_evaluation_loader(test_dataset)
        else:
            train_loader = get_evaluation_loader_MGCT(train_dataset)
            val_loader = get_evaluation_loader_MGCT(val_dataset)
            test_loader = get_evaluation_loader_MGCT(test_dataset)

        if args.model_type=='porpoise_mmf' or args.model_type=='early_mmf':
            args.omic_input_dim = train_dataset.genomic_features.shape[1]
            print('Omic input dim', args.omic_input_dim)
            model_dict = {'omic_input_dim': args.omic_input_dim, 'fusion': args.fusion, 'n_classes': args.n_classes, 
                          'gate_path': args.gate_path, 'gate_omic': args.gate_omic, 'scale_dim1': args.scale_dim1, 'scale_dim2': args.scale_dim2, 
                          'skip': args.skip, 'dropinput': args.dropinput, 'path_input_dim': args.path_input_dim, 'use_mlp': args.use_mlp,}

        elif args.model_type=='porpoise_amil':
            model_dict = {'n_classes': args.n_classes}

        elif args.model_type=='snn':
            args.omic_input_dim = train_dataset.genomic_features.shape[1]
            model_dict = {'omic_input_dim': args.omic_input_dim, 'model_size_omic': args.model_size_omic,
                          'n_classes': args.n_classes}

        elif args.model_type=='mgct':
            args.omic_sizes = [93, 329, 528, 457, 222, 447]
            args.omic_input_dim = 3189

            model_dict = {'fusion': args.fusion, 'omic_sizes': args.omic_sizes, 'n_classes': args.n_classes, 
                          'stage1_num_layers': 1, 'stage2_num_layers': 2}
            
        elif args.model_type=='mcat':           # falta revisar con su propio codigo
            args.omic_sizes = [93, 329, 528, 457, 222, 447]
            args.omic_input_dim = 3189 
            model_dict = {'fusion': args.fusion, 'omic_sizes': args.omic_sizes, 'n_classes': args.n_classes}

        results_train_dict, train_cindex = get_survival_results(train_loader, model_dict, args.model_type, i)
        results_val_dict, val_cindex = get_survival_results(val_loader, model_dict, args.model_type, i)
        results_test_dict, test_cindex = get_survival_results(test_loader, model_dict, args.model_type, i)

        latest_train_cindex.append(train_cindex)
        latest_val_cindex.append(val_cindex)
        latest_test_cindex.append(test_cindex)

        train_results_df = pd.DataFrame(results_train_dict)
        val_results_df = pd.DataFrame(results_val_dict)
        test_results_df = pd.DataFrame(results_test_dict)

        train_results_df.T.to_csv(os.path.join(args.results_dir, f'train_results_fold_{str(i)}.csv'))
        val_results_df.T.to_csv(os.path.join(args.results_dir, f'val_results_fold_{str(i)}.csv'))      # como evaluan en mgct y mcat
        test_results_df.T.to_csv(os.path.join(args.results_dir, f'test_results_fold_{str(i)}.csv'))

    results_sets_df = pd.DataFrame({'folds': folds, 'train_cindex': latest_train_cindex, 'val_cindex': latest_val_cindex, 'test_cindex': latest_test_cindex})

    mean_values = results_sets_df.iloc[:, 1:].mean()
    std_values = results_sets_df.iloc[:, 1:].std()

    results_sets_df = pd.concat([results_sets_df, pd.DataFrame([mean_values, std_values], index=['mean', 'std'])])

    results_sets_df.to_csv(os.path.join(args.results_dir, f'summary_final_{args.model_type}.csv'))

    print('Results')
    print(results_sets_df)

### Training settings
parser = argparse.ArgumentParser(description='Configurations for Survival Analysis on TCGA Data.')
### Checkpoint + Misc. Pathing Parameters
parser.add_argument('--data_root_dir',   type=str, default=EvaluateConfig.features_dir, help='Data directory to WSI features (extracted via CLAM')
parser.add_argument('--seed', 			 type=int, default=1, help='Random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', 			     type=int, default=5, help='Number of folds (default: 5)')
parser.add_argument('--k_start',		 type=int, default=-1, help='Start fold (Default: -1, last fold)')
parser.add_argument('--k_end',			 type=int, default=-1, help='End fold (Default: -1, first fold)')
parser.add_argument('--results_dir',     type=str, default='./results_new', help='Results directory (Default: ./results)')
parser.add_argument('--results_dir_evaluation',     type=str, default=EvaluateConfig.results_dir_evaluation, help='Results directory (Default: ./results)')
parser.add_argument('--which_splits',    type=str, default='5foldcv', help='Which splits folder to use in ./splits/ (Default: ./splits/5foldcv')
parser.add_argument('--split_dir',       type=str, default='tcga_blca', help='Which cancer type within ./splits/<which_splits> to use for training. Used synonymously for "task" (Default: tcga_blca_100)')
parser.add_argument('--log_data',        action='store_true', default=True, help='Log data using tensorboard')
parser.add_argument('--overwrite',     	 action='store_true', default=False, help='Whether or not to overwrite experiments (if already ran)')

### Model Parameters.
parser.add_argument('--model_type',      type=str, default='mcat', help='Type of model (Default: mcat)')
parser.add_argument('--mode',            type=str, choices=['omic', 'path', 'pathomic', 'pathomic_fast', 'cluster', 'coattn', 'early'], default='coattn', help='Specifies which modalities to use / collate function in dataloader.')
parser.add_argument('--fusion',          type=str, choices=['None', 'concat', 'bilinear'], default='None', help='Type of fusion. (Default: concat).')
parser.add_argument('--apply_sig',		 action='store_true', default=False, help='Use genomic features as signature embeddings.')
parser.add_argument('--apply_sigfeats',  action='store_true', default=False, help='Use genomic features as tabular features.')
parser.add_argument('--drop_out',        action='store_true', default=True, help='Enable dropout (p=0.25)')
parser.add_argument('--model_size_wsi',  type=str, default='small', help='Network size of AMIL model')
parser.add_argument('--model_size_omic', type=str, default='small', help='Network size of SNN model')

parser.add_argument('--n_classes', type=int, default=4)


### PORPOISE
parser.add_argument('--apply_mutsig', action='store_true', default=False)
parser.add_argument('--gate_path', action='store_true', default=False)
parser.add_argument('--gate_omic', action='store_true', default=False)
parser.add_argument('--scale_dim1', type=int, default=8)
parser.add_argument('--scale_dim2', type=int, default=8)
parser.add_argument('--skip', action='store_true', default=False)
parser.add_argument('--dropinput', type=float, default=0.0)
parser.add_argument('--path_input_dim', type=int, default=1024)
parser.add_argument('--use_mlp', action='store_true', default=False)


### Optimizer Parameters + Survival Loss Function
parser.add_argument('--opt',             type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--batch_size',      type=int, default=1, help='Batch Size (Default: 1, due to varying bag sizes)')
parser.add_argument('--gc',              type=int, default=32, help='Gradient Accumulation Step.')
parser.add_argument('--max_epochs',      type=int, default=20, help='Maximum number of epochs to train (default: 20)')
parser.add_argument('--lr',				 type=float, default=2e-4, help='Learning rate (default: 0.0001)')
parser.add_argument('--bag_loss',        type=str, choices=['svm', 'ce', 'ce_surv', 'nll_surv'], default='nll_surv', help='slide-level classification loss function (default: ce)')
parser.add_argument('--label_frac',      type=float, default=1.0, help='fraction of training labels (default: 1.0)')
parser.add_argument('--reg', 			 type=float, default=1e-5, help='L2-regularization weight decay (default: 1e-5)')
parser.add_argument('--alpha_surv',      type=float, default=0.0, help='How much to weigh uncensored patients')
parser.add_argument('--reg_type',        type=str, choices=['None', 'omic', 'pathomic'], default='None', help='Which network submodules to apply L1-Regularization (default: None)')
parser.add_argument('--lambda_reg',      type=float, default=1e-5, help='L1-Regularization Strength (Default 1e-4)')
parser.add_argument('--weighted_sample', action='store_true', default=True, help='Enable weighted sampling')
parser.add_argument('--early_stopping',  action='store_true', default=False, help='Enable early stopping')

### CLAM-Specific Parameters
parser.add_argument('--bag_weight',      type=float, default=0.7, help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--testing', 	 	 action='store_true', default=False, help='debugging tool')

args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoding_size = 1024
settings = {'num_splits': args.k, 
            'k_start': args.k_start,
            'k_end': args.k_end,
            'max_epochs': args.max_epochs, 
            'results_dir_evaluation': args.results_dir_evaluation,
            'results_dir': args.results_dir_evaluation,
            'lr': args.lr,
            'reg': args.reg,
            'label_frac': args.label_frac,
            'bag_loss': args.bag_loss,
            #'bag_weight': args.bag_weight,
            'seed': args.seed,
            'model_type': args.model_type,
            'model_size_wsi': args.model_size_wsi,
            'model_size_omic': args.model_size_omic,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'gc': args.gc,
            'opt': args.opt}
print('\nLoad Dataset')

args.n_classes = 4

if args.model_type=='porpoise_mmf' or args.model_type=='early_mmf':
    dataset = Generic_MIL_Survival_Dataset(csv_path = EvaluateConfig.csv_data_path,
                                            mode = args.mode,
                                            data_dir = EvaluateConfig.features_dir,
                                            shuffle = False, 
                                            seed = args.seed, 
                                            print_info = True,
                                            patient_strat= False,
                                            n_bins=4,
                                            label_col = 'survival_months',
                                            ignore=[])
else:
    dataset = Generic_MIL_Survival_Dataset_MGCT(csv_path = EvaluateConfig.csv_data_path,
                                            mode = args.mode,
                                            data_dir = EvaluateConfig.features_dir,
                                            shuffle = False, 
                                            seed = args.seed, 
                                            print_info = True,
                                            patient_strat= False,
                                            n_bins=4,
                                            label_col = 'survival_months',
                                            ignore=[])

# args.results_dir = args.results_dir_evaluation
args.results_dir = EvaluateConfig.results_dir_evaluation
args.results_dir_evaluation = EvaluateConfig.results_dir_evaluation
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

### Sets the absolute path of split_dir
args.split_dir = EvaluateConfig.splits_dir_survival
print("split_dir", args.split_dir)
assert os.path.isdir(args.split_dir)
settings.update({'split_dir': args.split_dir})

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))        

if __name__ == "__main__":
    results = main(args)
