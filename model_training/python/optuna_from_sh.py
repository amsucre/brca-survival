import argparse
import re
from pathlib import Path

from model_training import train_best_model
from optuna_grid_search import optuna_optimization, plot_study


# ===== MAIN =====
if __name__ == '__main__':
    # GET INPUT FROM USER
    # Define ArgumentParser object
    parser = argparse.ArgumentParser(description="Optuna neural network optimization")

    # Define the accepted arguments
    # parser.add_argument('-h', '--help', action='help', help="Show this help message and exit")
    parser.add_argument('-tr', '--inputtrain', required=True, help="Input TXT file for training", type=str)
    parser.add_argument('-te', '--inputtest', required=True, help="Input TXT file for testing", type=str)
    parser.add_argument('-nt', '--ntrial', required=True, help="Nº of trials", type=int)
    parser.add_argument('-ne', '--nepoch', required=True, help="Nº of epochs per trial", type=int)
    parser.add_argument('-nef', '--nepochfinal', required=True, help="Nº of epochs for final model", type=int)
    parser.add_argument('-nj', '--njobs', required=False, help="Nº of jobs (CPUs)", type=int, default=-1)  # -1 = Use all available cores
    parser.add_argument('-m', '--metric', required=False, help="Metric to optimize | Options: accuracy, loss", type=str, default='accuracy')
    parser.add_argument('-s', '--sampler', required=False, help="Optuna sampler (tpe or random)", type=str, default='tpe')
    parser.add_argument('-p', '--pruner', required=False, help="Optuna pruner (median, shp, none", type=str, default='none')
    parser.add_argument('-nls', '--nlayerstart', required=True, help="Nº of layers (Lower limit)", type=int)
    parser.add_argument('-nle', '--nlayerend', required=True, help="Nº of layers (Upper limit)", type=int)
    parser.add_argument('-nns', '--nneuronstart', required=True, help="Nº of neurons per layer (Lower limit)", type=int)
    parser.add_argument('-nne', '--nneuronend', required=True, help="Nº of neurons per layer (Upper limit)", type=int)
    parser.add_argument('-lrs', '--learningratestart', required=True, help="Learning rate (Lower limit)", type=float)
    parser.add_argument('-lre', '--learningrateend', required=True, help="Learning rate (Upper limit)", type=float)
    parser.add_argument('-bs', '--batchsizelist', required=True, help="Batch size options (dash-separated list)", type=str)
    parser.add_argument('-o', '--optimizerlist', required=True, help="Optimizer options (dash-separated list) | Available: Adam, SGD, SGD_M, RMSprop", type=str)
    parser.add_argument('-out', '--outputdir', required=False, help="Directory where to store optimization plots", type=str, default='')
    parser.add_argument('-l', '--loss', required=False, help="Define which loss function to use: default / porpoise", type=str, default='default')
    parser.add_argument('-as', '--alphastart', required=False, help="If using porpoise loss, define alpha range to test", type=str, default='')
    parser.add_argument('-ae', '--alphaend', required=False, help="If using porpoise loss, define alpha range to test", type=str, default='')
    parser.add_argument('-c', '--cindex', required=False, help="Define if we shoudl calculate or not the c_index for the models", type=str, default='')
    parser.add_argument('-e', '--early', required=False, help="Define if we should perform early stopping during training", type=str, default='')
    parser.add_argument('-smd', '--savemodeldir', required=False, help="Define folder where to store the best model (if needed)", type=str, default='')
    parser.add_argument('-spd', '--savepreddir', required=False, help="Define folder where to store the VAL quartiles predictions (if needed)", type=str, default='')
    parser.add_argument('-std', '--savepredtestdir', required=False, help="Define folder where to store the TEST quartiles predictions (if needed)", type=str, default='')
    parser.add_argument('-sdd', '--splitdatadir', required=False, help="If available, provide folder containing files splitting data in train/val", type=str, default='')
    parser.add_argument('-os', '--optunasplit', required=False, help="Indicate if 5-fold split shoudl be considered in Optuna or onyl in final training", type=str, default='')


    # Parse the given arguments
    args = parser.parse_args()
        
    # Input files
    train_fpath = args.inputtrain
    test_fpath = args.inputtest
    out_dir = args.outputdir

    # Input parameters
    n_trials = args.ntrial
    n_epoch = args.nepoch
    n_epoch_final = args.nepochfinal
    n_jobs = args.njobs
    metric = args.metric
    sampler = args.sampler
    pruner = args.pruner
    n_layer_range = (args.nlayerstart, args.nlayerend)
    n_neuron_range = (args.nneuronstart, args.nneuronend)
    lr_range = (args.learningratestart, args.learningrateend)
    batch_list = [int(x) for x in args.batchsizelist.split('-')]
    opt_list = args.optimizerlist.split('-')
    loss = args.loss
    
    a_start = args.alphastart
    a_end = args.alphaend
    alpha_range = (float(a_start), float(a_end)) if a_start and a_end else None
    
    get_c_index = True if args.cindex == 'T' else False
    early_stop = True if args.early == 'T' else False
    optuna_split = True if args.optunasplit == 'T' else False
    
    save_model_dir = args.savemodeldir
    save_pred_dir = args.savepreddir
    save_pred_test_dir = args.savepredtestdir
    split_dir = args.splitdatadir

    # MAIN FUNCTION
    # Run Optuna Optimization
    study = optuna_optimization(metric, n_trials, n_jobs, sampler, pruner, train_fpath, test_fpath, n_layer_range,
                                n_neuron_range, lr_range, batch_list, opt_list, n_epoch, loss, alpha_range, get_c_index, split_dir, optuna_split)
    best_trial = study.best_trial
    best_params = best_trial.params
    print('\nBest Hyperparameters:', best_params)

    # Result visualization
    plot_study(study, out_dir)

    # Train Best Model with Optimized Layers (single or many, based on input)
    print('Split: ', split_dir, Path(split_dir).is_dir())
    if split_dir and Path(split_dir).is_dir():
        print('Dir exists')
        for split_path in Path(split_dir).iterdir():
            print(f'==== {split_path.name} ====')
            n = int(re.search(r'\d+', split_path.name).group())
            save_model_path = Path(save_model_dir) / f'best_model_{n}.pth' if save_model_dir else None
            save_pred_path  = Path(save_pred_dir) / f'4Q_prediction_{n}.tsv' if save_pred_dir else None
            save_pred_test_path  = Path(save_pred_test_dir) / f'4Q_prediction_test_{n}.tsv' if save_pred_test_dir else None 
            outcome, y_pred, y_true, loss_per_epoch, c_index_dict, accuracy_per_epoch  = train_best_model(best_params, train_fpath, test_fpath, n_epoch_final, metric, loss, get_c_index, early_stop, save_model_path, save_pred_path, save_pred_test_path, split_path)
    
            # Best model results
            print(f'{metric.capitalize()}: {outcome:.4f}')
            print(f'C-Index: {c_index_dict['train']:.4f} (TR), {c_index_dict['val']:.4f} (VA), {c_index_dict['test']:.4f} (TE)')
            print(f'Loss/Epoch (Train): {loss_per_epoch['train']}')
            print(f'Loss/Epoch (Val): {loss_per_epoch['val']}')
            if metric == 'accuracy':
                print(f'Accuracy (Val): {accuracy_per_epoch['val']}')
            print('\n')
    else:
        split_path = None
        save_model_path = Path(save_model_dir) / 'best_model.pth' if save_model_dir else None
        save_pred_path  = Path(save_pred_dir) / '4Q_prediction.tsv' if save_pred_dir else None
        save_pred_test_path  = Path(save_pred_test_dir) / '4Q_prediction_test.tsv' if save_pred_test_dir else None 
        outcome, y_pred, y_true, loss_per_epoch, c_index_dict, accuracy_per_epoch  = train_best_model(best_params, train_fpath, test_fpath, n_epoch_final, metric, loss, get_c_index, early_stop, save_model_path, save_pred_path, save_pred_test_path, split_path)

        # Best model results
        print(f'{metric.capitalize()}: {outcome:.4f}')
        print(f'C-Index: {c_index_dict['train']:.4f} (TR), {c_index_dict['val']:.4f} (VA), {c_index_dict['test']:.4f} (TE)')
        print(f'Loss/Epoch (Train): {loss_per_epoch['train']}')
        print(f'Loss/Epoch (Val): {loss_per_epoch['val']}')
        if metric == 'accuracy':
            print(f'Accuracy (Val): {accuracy_per_epoch['val']}')
