import argparse

from model_training import train_best_model, plot_cmatrix


# ===== MAIN =====
if __name__ == '__main__':
    # GET INPUT FROM USER
    # Define ArgumentParser object
    parser = argparse.ArgumentParser(description="Train and test a given neural network")

    # Define the accepted arguments
    # parser.add_argument('-h', '--help', action='help', help="Show this help message and exit")
    parser.add_argument('-tr', '--inputtrain', required=True, help="Input TXT file for training", type=str)
    parser.add_argument('-te', '--inputtest', required=True, help="Input TXT file for testing", type=str)
    parser.add_argument('-ne', '--nepoch', required=True, help="Nº of epochs for final model", type=int)
    parser.add_argument('-nl', '--nlayer', required=True, help="Nº of layers", type=int)
    parser.add_argument('-nn', '--nneuronlist', required=True, help="Nº of neurons per layer (list)", type=str)
    parser.add_argument('-lr', '--learningrate', required=True, help="Learning rate", type=float)
    parser.add_argument('-bs', '--batchsize', required=True, help="Batch size", type=int)
    parser.add_argument('-o', '--optimizer', required=True, help="Optimizer", type=str)
    parser.add_argument('-out', '--outputdir', required=False, help="Directory where to store plots", type=str, default='')
    parser.add_argument('-l', '--loss', required=False, help="Define which loss function to use: default / porpoise", type=str, default='default')
    parser.add_argument('-m', '--metric', required=False, help="Metric to optimize | Options: accuracy, loss", type=str, default='accuracy')
    parser.add_argument('-a', '--alpha', required=False, help="If using porpoise loss, define alpha value", type=str, default='')
    parser.add_argument('-c', '--cindex', required=False, help="Define if we should calculate or not the c_index for the models", type=str, default='')
    parser.add_argument('-e', '--early', required=False, help="Define if we should perform early stopping during training", type=str, default='')
    parser.add_argument('-smp', '--savemodelpath', required=False, help="Define where to store the best model (if needed)", type=str, default='')
    parser.add_argument('-spp', '--savepredpath', required=False, help="Define where to store the VAL quartiles predictions (if needed)", type=str, default='')
    parser.add_argument('-stp', '--savepredtestpath', required=False, help="Define where to store the TEST quartiles predictions (if needed)", type=str, default='')
    parser.add_argument('-sdp', '--splitdatapath', required=False, help="If available, provide file splitting data in train/val", type=str, default='')

    # Parse the given arguments
    args = parser.parse_args()
        
    # Input files
    train_fpath = args.inputtrain
    test_fpath = args.inputtest
    out_dir = args.outputdir

    # Input parameters
    n_epoch = args.nepoch
    n_layer = args.nlayer
    n_neuron_list = [int(x) for x in args.nneuronlist.split('-')]
    lr = args.learningrate
    batch = args.batchsize
    opt = args.optimizer
    loss = args.loss
    metric = args.metric
    alpha = float(args.alpha) if args.alpha else 0.0
    get_c_index = True if args.cindex == 'T' else False
    early_stop = True if args.early == 'T' else False
    save_model_path = args.savemodelpath
    save_pred_path = args.savepredpath
    save_pred_test_path = args.savepredtestpath
    split_path = args.splitdatapath

    # MAIN FUNCTION
    # Define parameter dict
    best_params = {'num_layers': n_layer, 'lr': lr, 'batch_size': batch, 'optimizer': opt, 'alpha': alpha} | {f'hidden_size_{i}': x for i,x in enumerate(n_neuron_list)}    
    print('Best Hyperparameters:', best_params)

    # Train Best Model with Optimized Layers
    outcome, y_pred, y_true, loss_per_epoch, c_index_dict, accuracy_per_epoch = train_best_model(best_params, train_fpath, test_fpath, n_epoch, metric, loss, get_c_index, early_stop, save_model_path, save_pred_path, save_pred_test_path, split_path)

    # Best model results
    print(f'{metric.capitalize()}: {round(outcome, 5)}')
    print(f'C-Index: {c_index_dict['train']:.4f} (TR), {c_index_dict['val']:.4f} (VA), {c_index_dict['test']:.4f} (TE)')
    print(f'Loss/Epoch (Train): {loss_per_epoch['train']}')
    print(f'Loss/Epoch (Val): {loss_per_epoch['val']}')
    if metric == 'accuracy':
        print(f'Accuracy (Val): {accuracy_per_epoch['val']}')
      
    plot_cmatrix(y_true, y_pred, out_dir)
