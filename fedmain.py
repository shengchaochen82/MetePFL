import zipfile
import torch 
import random
from copy import deepcopy
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter
from aggregator import parameter_aggregate, read_out
from fed_utilis import *
from LocalUpdate import LocalClientUpdate
from base_module.data import data_split, generate_dataset, load_metr_la_data

from base_module.options import Options
from torch.utils.data import DataLoader, Dataset
from base_module.running import *
from base_module.data import *
from base_module.pretrain_trans import *
import os 

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def main(args):
    device = torch.device('cuda')
    logger.info("Using device: {}".format(device))
    if device == 'cuda':
        logger.info("Device index: {}".format(torch.cuda.current_device()))

    # Build data
    logger.info("Loading and preprocessing data ...")
    
    X, std, mean = load_metr_la_data(config['dataset'])

    logger.info('{} have been loader, the nodes is {}'.format(config['dataset'], X.shape[0]))

    # If graph attented
    A = np.zeros((config['clients'], config['clients']))

    my_data = X
    feat_dim = X.shape[1]  # dimensionality of data features NB 2

    # Split dataset

    split_line1 = int(X.shape[2] * 0.6)
    split_line2 = int(X.shape[2] * 0.8)

    train_original_data = my_data[:, :, :split_line1]
    val_original_data = my_data[:, :, split_line1:split_line2]
    test_original_data = my_data[:, :, split_line2:]

    train_indices = [i for i in range(train_original_data.shape[2] - config['total_len'])]
    test_indices = [i for i in range(test_original_data.shape[2] - config['total_len'])]
    val_indices = [i for i in range(val_original_data.shape[2] - config['total_len'])]

    train_set = generate_dataset(train_original_data, config['total_len'])
    val_set = generate_dataset(val_original_data, config['total_len'])
    test_set = generate_dataset(test_original_data, config['total_len'])

    logger.info("{} samples may be used for training".format(len(train_indices)))
    logger.info("{} samples will be used for validation".format(len(val_indices)))
    logger.info("{} samples will be used for testing".format(len(test_indices)))

    with open(os.path.join(config['output_dir'], 'data_indices.json'), 'w') as f:
        try:
            json.dump({'train_indices': list(map(int, train_indices)),
                       'val_indices': list(map(int, val_indices)),
                       'test_indices': list(map(int, test_indices))}, f, indent=4)
        except ValueError:  # in case indices are non-integers
            json.dump({'train_indices': list(train_original_data.shape[2]),
                       'val_indices': list(val_original_data.shape[2]),
                       'test_indices': list(test_original_data.shape[2])}, f, indent=4)

    # loading Pre-Trained Model
    model_dict = {'pro_len': config['prompting_length'], 'fore_len': config['forecasting_length'], 'ipt_len':config['input_length'], 
                  'pre_train': False, 'dataset': config['dataset'],
                  'whether_prompt': config['ynprompt'], 'Pretrained_II': True, 'prompt_app': config['prompt_app'], 'former_pretrain': config['former_pretrain']}


    logger.info("Creating model ...")
    logger.info("The {} has been loader.".format(model_dict['prompt_app']))


    if model_dict['whether_prompt'] == 'Normal_FL_Pretrain':
        model_dict['Pretrained_II'] = False
        model_dict['pre_train'] = True

        logger.info("Staring Federated Pre-Train")

    model = TSTransformerEncoder_Fed_Pre(feat_dim=feat_dim, max_len=config['input_length'] + config['forecasting_length'], d_model=config['d_model'], n_heads=config['num_heads'], 
                                         num_layers=config['num_layers'], dim_feedforward=config['dim_feedforward'], model_dict=model_dict).cuda()
    model.train()
    freezex(layer_name='Transformer_backbone', model=model)
    # freezex(layer_name='Transformer_prompt_pre', model=model)


    logger.info("Model:\n{}".format(model))
    logger.info("Total number of parameters: {}".format(count_parameters(model)))
    logger.info("Trainable parameters: {}".format(count_parameters(model, trainable=True)))

    # Federated Setting
    w_server, w_local = model.get_state()

    w_server = [w_server] * config['clients']
    w_local = [w_local] * config['clients']

    global_model = deepcopy(w_server)
    personalized_model = deepcopy(w_server)

    server_state = None

    # Tensorborad Staring
    communication_board = SummaryWriter(config['tensorboard_dir'])
    # Dataset Preparing
    num_collaborator = max(int(config['client_frac'] * config['clients']), 1)
    dict_user = data_split(config['nodes'], config['clients'])

    fed_dict = {'lr': config['lr'], 'tensorboard_dir': config['tensorboard_dir'], 'batch_size': config['batch_size'],
                'num_workers': config['num_workers'], 'device': "cuda", 'print_interval': config['print_interval'],
                'console': config['console'], 'start_epoch': 0, 'epochs': config['epochs'], 'valid_fre': 5,
                'masking_ratio': config['masking_ratio'], 'mean_mask_length': config['mean_mask_length'],
                'input_len': config['input_length'], 'forecasting_len': config['forecasting_length']                                                                                                                                                                                                                                                                                                                                                                                          ,
                'prompt_len': config['prompting_length'], 'd_model': config['d_model'], 'dim_feedforward': config['dim_feedforward'],
                'num_heads': config['num_heads'], 'num_layers': config['num_layers'], 'feat_dim': feat_dim}


    agg_dict = {'agg_app': config['agg'], 'clients': config['clients'], 'sub_graph': config['clients'],
                'serverlpha': 0.3, 'adjbeta': 0.7}

  
    # Federated Learning Training 
    if model_dict['whether_prompt'] == 'Prompt_learning_interact' or model_dict['whether_prompt'] == 'normal_unsup' or model_dict['whether_prompt'] == "Novel_Prompting" or model_dict['whether_prompt'] == "Normal_Prompting":
        logger.info('Dataset Remove')
        train_set = val_set
        train_indices = val_indices
        val_set = test_set
        val_indices = test_indices
    
    balance_martix = torch.zeros(config['clients'], config['clients'])
    for com in range(1, config['com_round'] + 1):
        selected_user = np.random.choice(range(config['clients']), num_collaborator, replace=False)
        train_time = []
        train_loss = []
        train_mae = []
        train_rmse = []

        client_recoder = []

        for c in selected_user:
            client_recoder.append(c)
            engine = LocalClientUpdate(config, dict_user[c], train_set, train_indices, 
                                       val_set, val_indices, global_model[c], personalized_model[c],
                                          w_local[c], {}, c, 0, config['local_mode_t'], server_state, mean, std, fed_dict, model_dict)
            outputs = engine.run()

            w_server[c] = deepcopy(outputs['params'][0])
            w_local[c] = deepcopy(outputs['params'][1])
            train_time.append(outputs["time"])
            train_loss.append(outputs["loss"])
            train_mae.append(outputs["mae"])
            train_rmse.append(outputs['rmse'])
            communication_board.add_scalar('Client_Training:{}'.format(c), train_mae[-1], com)


        mtrain_time = np.mean(train_time)
        mtrain_loss = np.mean(train_loss)
        mtrain_mae = np.mean(train_mae)
        mtrain_rmse = np.mean(train_rmse)

        communication_board.add_scalar('Communication Round:{}'.format(com), mtrain_mae, com)


        logger.info('Communication Round: {}, Train Loss: {},'\
            ' Train MSE/RMSE: {}, {}, Training Time: {}/com_round'.format(com, mtrain_loss, mtrain_mae, mtrain_rmse, mtrain_time))

        
        logger.info('----- Staring Aggregation ------')

        t1 = time.time()
        personalized_model = parameter_aggregate(args, A, w_server, global_model, agg_dict, client_recoder, balance_martix)
        t2 = time.time()

        logger.info('Communication Round: {}, Aggregation Time: {}'.format(com, (t2 - t1)))

        # global_model = personalized_model
        global_model = read_out(personalized_model, "cuda")

        logger.info('----- Staring validation round ------')

        if com % fed_dict['valid_fre'] == 0:

            all_vtime = []
            all_vloss = []
            all_vacc = []
            all_vrmse = []

            best_metrics = {'best_mae': 0, 'best_rmse': 0}

            batch_time = []
            batch_loss = []
            batch_mae = []
            batch_rmse = []

            for c in range(config['clients']):

                tengine = LocalClientUpdate(args, dict_user[c], [], [], 
                                            val_set, val_indices, personalized_model[c], personalized_model[c],
                                               w_local[c], {}, c, 0, config['local_mode_v'], server_state, mean, std, fed_dict, model_dict=model_dict)
                outputs = tengine.run()

                batch_time.append(outputs["time"])
                batch_loss.append(outputs["loss"])
                batch_mae.append(outputs["mae"])
                batch_rmse.append(outputs['rmse'])
                communication_board.add_scalar('Client_Validation:{}'.format(c), train_mae[-1], c)

            all_vtime.append(np.mean(batch_time))
            all_vloss.append(np.mean(batch_loss))
            all_vacc.append(np.mean(batch_mae))
            all_vrmse.append(np.mean(batch_rmse))



            logger.info('AllValidation Round: {}, Valid Loss: {}, ' \
                         'Valid MAE/RMSE: {},{}, Valid SD: {}, Test Time: {}/epoch'.
                         format(com, np.mean(all_vloss), np.mean(all_vacc), np.mean(all_vrmse), np.std(all_vacc),
                                      np.mean(all_vtime)))

            
            best_metrics['best_mae'], best_metrics['best_rmse'] = np.mean(all_vacc), np.mean(all_vrmse)
            save_model(os.path.join(config['save_dir'], 'model_{}.pth'.format('best')), epoch=com, model = model, optimizer=None)
            logger.info("Best Model has been saved ")
            
            logger.info('Best MAE: {}, Best RMSE: {}'.format(best_metrics['best_mae'], best_metrics['best_rmse']))


if __name__ == "__main__":
    args = Options().parse()  # `argsparse` object
    config = setup(args)  # configuration dictionary
    main(config)