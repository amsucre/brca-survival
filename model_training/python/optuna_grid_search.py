import multiprocessing
from pathlib import Path

import optuna
import optuna.visualization as vis
import torch

from common_utils import BRCAClassifier, BRCADataset, load_data, train_and_evaluate_model_accuracy, train_and_evaluate_model_loss


# ===== MAIN FUNCTIONS =====
def optuna_optimization(how: str, n_trials: int, n_jobs: int, sampler: str, pruner: str, train_fpath: str,
                        test_fpath: str, n_layer_range: tuple[int, int], n_neuron_range: tuple[int, int],
                        lr_range: tuple[float, float], batch_list: list[int], opt_list: list[str], n_epoch: int, 
                        which_loss: str = 'default', alpha_range: tuple[int, int] | None = None, get_c_index: bool = False, split_dir: str = '', optuna_split: bool = False):
    
    # === OPTUNA STUDY PARAMETERS ===
    # DIRECTION > maximize (Enlarge accuracy, F1, precision, recall)
    #           > minimize (Reduce loss function)
    # SAMPLER > TPESampler (Default. Bayesian optimization. Suggests new hyperparameters based on previous iterations)
    #         > RandomSampler (The set of hyperparameters are randomly chosen from the defined search space)
    #         > GridSampler (The set of hyperparameters are chosen based on every combination in the search space)
    #         > There are other options. Pending to explore them
    # PRUNER > MedianPruner (Safer), SuccessiveHalvingPruner (Faster), PatientPruner, PercentilePruner...

    # Define parameters based on input
    if how == 'accuracy':
        direction = 'maximize'
    elif how == 'loss':
        direction = 'minimize'
    else:
        return

    sampler_dict = {'tpe': optuna.samplers.TPESampler(),
                    'random': optuna.samplers.RandomSampler(),
                    'none': None}
    sampler = sampler if sampler in sampler_dict else 'tpe'  # Default

    pruner_dict = {'median': optuna.pruners.MedianPruner(n_startup_trials=max(3, int(n_trials/100)),
                                                         n_warmup_steps=max(3, int(n_epoch/5))),
                   'shp': optuna.pruners.SuccessiveHalvingPruner(),   # TODO: Definir parametros ideales
                   'none': None
                   }
    pruner = pruner if pruner in pruner_dict else 'none'

    # Run optuna optimization (usamos lambda para poder pasar mas parametros) || Process varies depending on input (using splits or not??)
    study = optuna.create_study(direction=direction, sampler=sampler_dict[sampler], pruner=pruner_dict[pruner])
    study.optimize(lambda trial: objective(trial, how, n_layer_range, n_neuron_range, lr_range, batch_list, opt_list,
                                           train_fpath, test_fpath, n_epoch, pruner != 'none', which_loss, alpha_range, get_c_index, split_dir, optuna_split),
                   n_trials=n_trials, n_jobs=n_jobs, gc_after_trial=True)

    print("Final check for stuck processes...")   
    print(multiprocessing.active_children())

    return study


# Optuna Objective Function
def objective(trial: optuna.Trial, how: str, n_layer_range: tuple[int, int], n_neuron_range: tuple[int, int],
              lr_range: tuple[float, float], batch_list: list[int], opt_list: list[str], train_fpath: str,
              test_fpath: str, n_epoch: int, has_pruner: bool, which_loss: str, alpha_range: tuple[int, int] | None, 
              get_c_index: bool, split_dir: str, optuna_split: bool) -> float:
    try:
        optuna.logging.get_logger('optuna').info(f"TRIAL {trial.number} STARTED...")  
    
        # Suggest model parameters: number of layers, neurons per layer, learning rate, batch size & optimizer
        num_layers = trial.suggest_int('num_layers', *n_layer_range)
        parameters = {'num_layers': num_layers,
                      'hidden_sizes': [trial.suggest_int(f'hidden_size_{i}', *n_neuron_range, step=4) for i in range(num_layers)],
                      'lr': trial.suggest_float('lr', *lr_range, log=True),
                      'batch_size': trial.suggest_categorical('batch_size', batch_list),
                      'optimizer': trial.suggest_categorical('optimizer', opt_list)
                      }
        
        if which_loss == 'porpoise' and alpha_range:
            parameters['alpha'] =  trial.suggest_float('alpha', *alpha_range, step=0.1)

        # Prep data
        x_train, y_train, t_train, c_train, s_train = load_data(train_fpath)
        # x_test, y_test, t_test, c_test, s_test = load_data(test_fpath)
        dataset_train = BRCADataset(x_train, y_train, t_train, c_train, s_train)
        # dataset_test = BRCADataset(x_test, y_test, t_test, c_test, s_test)

        # Define model
        model = BRCAClassifier(x_train.shape[1], parameters)
        
        # Train and optimize models, depending if a valid split_dir was given or not
        if optuna_split and split_dir and Path(split_dir).is_dir():  # Train n models and get average parameter (value to optimize)
            outcome_list = list()
            n = 1
            for split_path in Path(split_dir).iterdir():
                optuna.logging.get_logger('optuna').info(f"Running model #{n} using {split_path.name}")
                n += 1
                if how == 'accuracy':
                    outcome_list.append(train_and_evaluate_model_accuracy(model, parameters, dataset_train, None, n_epoch, 
                                                                          which_loss, has_pruner=has_pruner, trial=trial, 
                                                                          timeout=True, get_c_index=get_c_index, split_path=split_path))
                elif how == 'loss':
                    outcome_list.append(train_and_evaluate_model_loss(model, parameters, dataset_train, None, n_epoch, 
                                                                      which_loss, has_pruner=has_pruner, trial=trial, 
                                                                      timeout=True, get_c_index=get_c_index, split_path=split_path))
                else: 
                    break
            
            outcome = sum(outcome_list)/len(outcome_list) if len(outcome_list) else None
      
        else:  # Train single model and get single parameter (value to optimize) 
            if how == 'accuracy':
                outcome = train_and_evaluate_model_accuracy(model, parameters, dataset_train, None, n_epoch, which_loss,
                                                            has_pruner=has_pruner, trial=trial, timeout=True, get_c_index=get_c_index)
            elif how == 'loss':
                outcome = train_and_evaluate_model_loss(model, parameters, dataset_train, None, n_epoch, which_loss,
                                                        has_pruner=has_pruner, trial=trial, timeout=True, get_c_index=get_c_index)
            else: 
                outcome = None
                                            
        # Free memory
        torch.cuda.empty_cache()    # clear torch cache
        
        return outcome
    except RuntimeError as e:
        print(f"Error in trial {trial.number}: {repr(e)}, {e}")
        return None
    

def plot_study(study: optuna.Study, out_dir: str = None):
    # 1. Plot Optimization History (Accuracy over Trials)
    fig1 = vis.plot_optimization_history(study)
 
    # 2. Plot Hyperparameter Importance (Which parameters matter most?)
    fig2 = vis.plot_param_importances(study)

    # 3. Contour Plot (Shows how two parameters interact)
    fig3 = vis.plot_contour(study, params=["hidden_size_0", "hidden_size_1"])

    # 4. Parallel Coordinate Plot (Shows relationships between multiple parameters)
    fig4 = vis.plot_parallel_coordinate(study)

    # 5. Slice Plot (Shows value distributions of each parameter)
    fig5 = vis.plot_slice(study)
    
    if out_dir:
        out_dir_obj = Path(out_dir)
        out_dir_obj.mkdir(parents=True, exist_ok=True)
        fig1.write_image(out_dir_obj / 'opt_history.png')
        fig2.write_image(out_dir_obj / 'param_importance.png')
        fig3.write_image(out_dir_obj / 'contour.png')
        fig4.write_image(out_dir_obj / 'parallel_coord.png')
        fig5.write_image(out_dir_obj / 'slice.png')
    else:
        fig1.show()
        fig2.show()
        fig3.show()
        fig4.show()
        fig5.show()
