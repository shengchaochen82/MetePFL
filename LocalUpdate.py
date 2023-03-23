import datetime
import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
from fed_utilis import PiecewiseLinear, StatsLogger, sd_matrixing, trainable_params
import time
from base_module.dataset import *
from base_module.options import Options
from base_module.running import *
from base_module.loss import *
import tqdm
from torch.utils.tensorboard import SummaryWriter
from base_module.promp_mask import *
from base_module.pretrain_trans import *

from Transformer_trainer import *

class LocalClientUpdate:
    def __init__(self, args, dict_user, train_dataset, train_indices, val_dataset, val_indices, global_param, server_param, local_param,
                 outputs, cid, tid, mode, server_state, means, stds, client_dict, model_dict):
        self.args = args
        self.dict_user = dict_user
        self.client_dict = client_dict
        self.model_dict = model_dict

        self.global_param = global_param
        self.server_param = server_param
        self.local_param = local_param
        self.server_state = server_state
        self.client_id = cid
        self.outputs = outputs
        self.thread = tid
        self.mode = mode

        self.train_set = train_dataset
        self.train_indices = train_indices

        self.val_set = val_dataset
        self.val_indices = val_indices
        

        self.model = self.prepare_model()
        

        self.loss_module = MaskedMSELoss(reduction='none')


        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=client_dict['lr'])
        self.means = means
        self.stds = stds

        self.tensorboard_writer = SummaryWriter(client_dict['tensorboard_dir'])

        if self.mode == "Train":
            if self.model_dict['Pretrained_II'] == False:
                self.train_dataset = ImputationDataset(self.train_set[:, list(self.dict_user), :, :], self.train_indices, masking_ratio = client_dict['masking_ratio'], 
                                                            mean_mask_length=client_dict['mean_mask_length'])
            else:
                self.train_dataset = Imputation_Inter_Prompting_Dataset(self.train_set[:, list(self.dict_user), :, :], self.train_indices, client_dict['masking_ratio'], 
                                                                client_dict['mean_mask_length'], client_dict['input_len'], client_dict['forecasting_len'], 
                                                                client_dict['prompt_len'], mode="Data_Preparing")
            self.train_loader = DataLoader(
                dataset=self.train_dataset,
                batch_size=client_dict['batch_size'],
                shuffle=False,
                num_workers=client_dict['num_workers'],
                pin_memory=True,
                collate_fn=lambda x : collate_unsuperv(x, max_len=self.model.max_len),
                drop_last=True
            )
            self.trainer = UnsupervisedRunner(self.model, self.train_loader, client_dict['device'], self.loss_module, self.optimizer,
                                    print_interval=client_dict['print_interval'], console=client_dict['console'], model_dict=model_dict, 
                                    global_param = self.global_param, local_param = self.local_param, server_param = self.server_param)

        elif self.mode == "Test":
            if self.model_dict['Pretrained_II'] == False:
                self.val_dataset = ImputationDataset(self.val_set[:, list(self.dict_user), : ,:], self.val_indices, masking_ratio = client_dict['masking_ratio'], 
                                                            mean_mask_length=client_dict['mean_mask_length'])

            else:
                self.val_dataset = Imputation_Inter_Prompting_Dataset(self.val_set[:, list(self.dict_user), : ,:], self.val_indices, client_dict['masking_ratio'], 
                                                                client_dict['mean_mask_length'], client_dict['input_len'], client_dict['forecasting_len'], 
                                                                client_dict['prompt_len'], mode="Data_Preparing")

            self.val_loader = DataLoader(dataset=self.val_dataset,
                                batch_size=client_dict['batch_size'],
                                shuffle=False,
                                num_workers=client_dict['num_workers'],
                                pin_memory=True,
                                collate_fn=lambda x: collate_unsuperv(x, max_len=self.model.max_len),
                                drop_last=True
                                )
                                
            self.val_evaluator = UnsupervisedRunner(self.model, self.val_loader, client_dict['device'], self.loss_module,
                                       print_interval=client_dict['print_interval'], console=client_dict['console'], model_dict=model_dict)


    def prepare_model(self):
        # TODO: pre-trained model with a trainable layers
        model = TSTransformerEncoder_Fed_Pre(feat_dim=self.client_dict['feat_dim'], max_len=32, d_model=self.client_dict['d_model'],
                                             n_heads=self.client_dict['num_heads'], num_layers=self.client_dict['num_layers'], 
                                             dim_feedforward=self.client_dict['dim_feedforward'], model_dict=self.model_dict).cuda()
        model.set_state(self.global_param, self.local_param)

        if self.model_dict['whether_prompt'] != 'Normal_FL_Pretrain':
            logger.info("BLOCK-----------------BLOCK")
            freezex(layer_name='Transformer_backbone', model=model)
            freezex(layer_name='Transformer_prompt_pre', model=model)
            # freezex(layer_name='prompt_en', model=model)
        return model

    def run(self):

        mean_mae = []
        mean_loss = []
        mean_rmse = []
        t1 = time.time()
        if self.mode == "Train":
            logger.info("starting training....")
            total_epoch_time = .0
            for epoch in tqdm.tqdm(range(self.client_dict['start_epoch'] + 1, self.client_dict["epochs"] + 1), desc='Training Epoch', leave=False):
                epoch_start_time = time.time()
                aggr_metrics_train = self.trainer.train_epoch(epoch)  # dictionary of aggregate epoch metrics
                epoch_runtime = time.time() - epoch_start_time

                mean_mae.append(aggr_metrics_train['mae'])
                mean_loss.append(aggr_metrics_train['loss'])
                mean_rmse.append(aggr_metrics_train['rmse'])

                print_str = 'Epoch {} Training Summary: '.format(epoch)
                for k, v in aggr_metrics_train.items():
                    self.tensorboard_writer.add_scalar('Client{}/{}/train'.format(self.client_id, k), v, epoch)
                    print_str += '{}: {:8f} | '.format(k, v)
                total_epoch_time += epoch_runtime
                avg_epoch_time = total_epoch_time / (epoch - self.client_dict['start_epoch'])
                avg_batch_time = avg_epoch_time / len(self.train_loader)
                avg_sample_time = avg_epoch_time / len(self.train_loader)



        elif self.mode == "Test":
            logger.info("starting validating....")
            epoch_start_time = time.time()
            with torch.no_grad():
                aggr_metrics_train = self.val_evaluator.evaluate()  # dictionary of aggregate epoch metrics
            epoch_runtime = time.time() - epoch_start_time

            mean_mae.append(aggr_metrics_train['mae'])
            mean_loss.append(aggr_metrics_train['loss'])
            mean_rmse.append(aggr_metrics_train['rmse'])

            logger.info("Epoch runtime: {} hours, {} minutes, {} seconds\n".format(*readable_time(epoch_runtime)))

        elif self.mode == "Informer_train" or self.mode == "Autoformer_train" or self.mode == "Fedformer_train":
            logger.info('Staring {} that is pretty simple'.format(self.mode))
            total_epoch_time = .0
            for epoch in tqdm.tqdm(range(self.client_dict['start_epoch'] + 1, self.client_dict["epochs"] + 1), desc='Training Epoch', leave=False):
                epoch_start_time = time.time()
                
                aggr_metrics_train = train_fedformer_epoch(self.model, self.optimizer, self.loss_module, self.rmse_cri, training_sequence=self.train_colset, num_ipt=self.client_dict['input_len'],
                                                            num_pre=self.client_dict['forecasting_len'] + self.client_dict['prompt_len'] - self.client_dict['input_len'], batch_size=128, model_dict=self.model_dict)

                epoch_runtime = time.time() - epoch_start_time

                mean_mae.append(aggr_metrics_train['mae'])
                mean_loss.append(aggr_metrics_train['loss'])
                mean_rmse.append(aggr_metrics_train['rmse'])

                print_str = 'Epoch {} Training Summary: '.format(epoch)
                for k, v in aggr_metrics_train.items():
                    self.tensorboard_writer.add_scalar('Client{}/{}/train'.format(self.client_id, k), v, epoch)
                    print_str += '{}: {:8f} | '.format(k, v)

                total_epoch_time += epoch_runtime
                avg_epoch_time = total_epoch_time / (epoch - self.client_dict['start_epoch'])

        
        elif self.mode == "Informer_val" or self.mode == "Autoformer_val" or self.mode == "Fedformer_val":
            logger.info('Staring {} that is pretty simple'.format(self.mode))
            epoch_start_time = time.time()
            with torch.no_grad():
                aggr_metrics_val = val_fedformer_epoch(self.model, self.loss_module, self.rmse_cri, self.val_colset, num_ipt=self.client_dict['input_len'],
                                                        num_pre=self.client_dict['forecasting_len'] + self.client_dict['prompt_len'] - self.client_dict['input_len'], batch_size=128, model_dict=self.model_dict)
            epoch_runtime = time.time() - epoch_start_time
            mean_mae.append(aggr_metrics_val['mae'])
            mean_loss.append(aggr_metrics_val['loss'])
            mean_rmse.append(aggr_metrics_val['rmse'])
            
            logger.info("Epoch runtime: {} hours, {} minutes, {} seconds\n".format(*readable_time(epoch_runtime)))


        time_cost = time.time() - t1


        if self.mode == "Test" or self.mode == "Informer_val" or self.mode == "Autoformer_val" or self.mode == "Fedformer_val":
            logger.info('Client:{}. Average Loss:{},' \
                        ' Average MAE/RMSE: {}/{}, Total Time Cost: {}'.format(self.client_id, 
                                                                        np.mean(mean_loss), np.mean(mean_mae),np.mean(mean_rmse),
                                                                                        time_cost))
            output = {"params": self.model.get_state(),
                        "time": time_cost,
                        "loss": np.mean(mean_loss),
                        "mae": np.mean(mean_mae),
                        "rmse":np.mean(mean_rmse)}

        elif self.mode == "Train" or self.mode == "Informer_train" or self.mode == "Autoformer_train" or self.mode == "Fedformer_train":
            logger.info('Client:{}. Average Loss:{},' \
                        ' Average MAE: {}, Loss:{}, MAE:{}, RMSE:{}, Total Time Cost: {}'.format(self.client_id, 
                        np.mean(mean_loss), np.mean(mean_mae), aggr_metrics_train['loss'], aggr_metrics_train['mae'], 
                        aggr_metrics_train['rmse'], time_cost))

            output = {"params": self.model.get_state(),
                        "time": time_cost,
                        "loss": aggr_metrics_train['loss'],
                        "mae": aggr_metrics_train['mae'],
                        'rmse': aggr_metrics_train['rmse']}

        return output

if __name__ == '__main__':
    args = Options().parse()
    config = setup(args)
    LocalClientUpdate(config)
