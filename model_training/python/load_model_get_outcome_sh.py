import argparse
import sys

from common_utils import load_model_and_get_outcomes



# ===== MAIN =====
if __name__ == '__main__':
    # GET INPUT FROM USER
    # Define ArgumentParser object
    parser = argparse.ArgumentParser(description="Load a model from a PTH file and get certain data")

    # Define the accepted arguments
    parser.add_argument('-mp', '--modelpath', required=True, help="Path to PTH file where model is stored", type=str)
    
    parser.add_argument('-tdp', '--testdatapath', required=False, help="Path to the data file (Test)", type=str, default='')
    parser.add_argument('-vdp', '--valdatapath', required=False, help="Path to the data file (Validation)", type=str, default='')
    parser.add_argument('-sdp', '--splitdatapath', required=False, help="Path to the data file (Split)", type=str, default='')    
    
    parser.add_argument('-vqp', '--valqpath', required=False, help="Path where to store the 4 quartiles prediction (Validation)", type=str, default='')
    parser.add_argument('-vcp', '--valclasspath', required=False, help="Path where to store the class prediction (Validation)", type=str, default='')
    parser.add_argument('-vrp', '--valriskpath', required=False, help="Path where to store the global risk prediction (Validation)", type=str, default='')
    parser.add_argument('-vfp', '--valfinalpath', required=False, help="Path where to store the activation data from the final hidden layer (Validation)", type=str, default='')
    parser.add_argument('-vqrp', '--valqrpath', required=False, help="Path where to store the 4 quartiles RAW prediction (Validation)", type=str, default='')

    parser.add_argument('-tqp', '--testqpath', required=False, help="Path where to store the 4 quartiles prediction (Test)", type=str, default='')
    parser.add_argument('-tcp', '--testclasspath', required=False, help="Path where to store the class prediction (Test)", type=str, default='')
    parser.add_argument('-trp', '--testriskpath', required=False, help="Path where to store the global risk prediction (Test)", type=str, default='')
    parser.add_argument('-tfp', '--testfinalpath', required=False, help="Path where to store the activation data from the final hidden layer (Test)", type=str, default='')
    parser.add_argument('-tqrp', '--testqrpath', required=False, help="Path where to store the 4 quartiles RAW prediction (Test)", type=str, default='')
    
    parser.add_argument('-ms', '--modelsize', required=False, help="If not in PTH, provide the model input size", type=int, default=0)
    parser.add_argument('-ml', '--modelloss', required=False, help="If not in PTH, provide the model loss function", type=str, default='')
    
    parser.add_argument('-nl', '--nlayer', required=False, help="Nº of layers", type=int)
    parser.add_argument('-nn', '--nneuronlist', required=False, help="Nº of neurons per layer (list)", type=str)
    parser.add_argument('-lr', '--learningrate', required=False, help="Learning rate", type=float)
    parser.add_argument('-bs', '--batchsize', required=False, help="Batch size", type=int)
    parser.add_argument('-o', '--optimizer', required=False, help="Optimizer", type=str)
    parser.add_argument('-a', '--alpha', required=False, help="If using porpoise loss, define alpha value", type=str, default='')

    # Parse the given arguments
    args = parser.parse_args()
        
    # Input files
    model_fpath = args.modelpath
    test_fpath = args.testdatapath
    val_fpath = args.valdatapath
    split_fpath = args.splitdatapath
    
    # Output files
    val_q_fpath = args.valqpath
    val_class_fpath = args.valclasspath
    val_risk_fpath = args.valriskpath
    val_final_fpath = args.valfinalpath
    val_q_raw_fpath = args.valqrpath

    test_q_fpath = args.testqpath
    test_class_fpath = args.testclasspath
    test_risk_fpath = args.testriskpath
    test_final_fpath = args.testfinalpath
    test_q_raw_fpath = args.testqrpath
    
    # Parameters
    model_size = args.modelsize
    model_loss = args.modelloss
    
    n_layer = args.nlayer
    n_neuron_list = [int(x) for x in args.nneuronlist.split('-')]
    lr = args.learningrate
    batch = args.batchsize
    opt = args.optimizer
    alpha = float(args.alpha) if args.alpha else 0.0
    
    # Define aux parameters from input
    outcome_dict = {'val_prob_4q': val_q_fpath, 
                    'val_pred_class': val_class_fpath, 
                    'val_risk': val_risk_fpath, 
                    'val_final_hidden': val_final_fpath,
                    'val_4q_raw': val_q_raw_fpath,
                    'test_prob_4q': test_q_fpath, 
                    'test_pred_class': test_class_fpath, 
                    'test_risk': test_risk_fpath, 
                    'test_final_hidden': test_final_fpath,
                    'test_4q_raw': test_q_raw_fpath,
                    }
    outcome_dict = {k: v for k, v in outcome_dict.items() if v != ''}
    
    if model_size > 0:   # If parameters are given, then we know they are not store in the model pth file
        aux_in_file = False
        
        # Prep model_param_dict
        model_param_dict = {'num_layers': n_layer, 'lr': lr, 'batch_size': batch, 'optimizer': opt, 'alpha': alpha} | {f'hidden_size_{i}': x for i,x in enumerate(n_neuron_list)}
    else:
        aux_in_file = True
        model_size = None
        model_loss = None
        model_param_dict = None
     
    # Check if input REQUIRED data is given   
    if not model_fpath and not model_fpath.endswith('.pth'):
        print('ERROR: You must provide a .PTH file where your model is stored')
        sys.exit()
        
    if not (test_fpath or val_fpath):
        print('ERROR: You must provide at least one data file: Validation or Testing')
        sys.exit()

    # MAIN FUNCTION
    load_model_and_get_outcomes(model_fpath, outcome_dict, aux_in_file, model_size, model_param_dict, model_loss, test_fpath, val_fpath, split_fpath)