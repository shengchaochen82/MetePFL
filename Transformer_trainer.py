from collections import OrderedDict
from sklearn import metrics
import torch
import numpy as np
import torch.nn as nn
import copy
from basemodule.promp_mask import geom_noise_mask_single
from basemodule.dataset import noise_mask
from basemodule.loss import MaskedMSELoss, MaskedRMSE


def train_fedformer_epoch(net, optimizer, loss_criterion, rmse_cri, training_sequence, num_ipt, num_pre, batch_size, model_dict):

    permutation = torch.randperm(training_sequence.shape[0])

    epoch_training_losses = []
    epoch_training_mae = []
    epoch_training_rmse = []
    rmse_train = 0
    epoch_metrics = OrderedDict()

    for i in range(0, (training_sequence.shape[0] // batch_size) * batch_size, batch_size):
        net.train()
        optimizer.zero_grad()
        indices = permutation[i:i + batch_size]
        mask_enc_co = []

        X_batch = training_sequence[indices]
        # (128, 30, 12) (128, 18)
        X_batch = X_batch.squeeze()
        ground = copy.deepcopy(X_batch)[:,  num_ipt:, -1]
        
        enc_ipt = X_batch
        enc_mask = copy.copy(enc_ipt)[:, :, :5]

        if model_dict['former_pretrain'] != True and model_dict['whether_prompt'] == "No_Former":
            # print("ssss")
            dec_ipt = X_batch[:, -num_pre:, :]
            dec_ipt = np.concatenate([dec_ipt, np.zeros_like(dec_ipt[:, :, :])], axis=1)

            dec_mask = copy.deepcopy(dec_ipt)[:, :, :5]
            enc_ipt, enc_mask, dec_ipt, dec_mask = torch.from_numpy(enc_ipt).cuda(), torch.from_numpy(enc_mask).cuda(), \
                                                        torch.from_numpy(dec_ipt).cuda(), torch.from_numpy(dec_mask).cuda()
            prediction = net(enc_ipt, enc_mask, dec_ipt, dec_mask)
            z = prediction.squeeze()
            ground = torch.from_numpy(ground).cuda()
            loss = loss_criterion(ground, z)
        else:
            if model_dict['whether_prompt'] == "normal_unsup" or model_dict['whether_prompt'] == "Novel_Prompting" or model_dict['whether_prompt'] == "Normal_Prompting":
                for i in range(len(enc_ipt)):
                    mask_enc = torch.ones((27, 12), dtype=bool).cuda()
                    mask_enc[:model_dict['ipt_len'], :] = torch.zeros_like(mask_enc[:model_dict['ipt_len'], :], dtype=bool)
                    mask_enc_co.append(mask_enc)
            else:
                for i in range(len(enc_ipt)):
                    mask_enc =  noise_mask(enc_ipt[i], masking_ratio=0.15, lm=3)
                    mask_enc = torch.from_numpy(mask_enc).cuda()
                    mask_enc_co.append(mask_enc)

            mask_enc_co = torch.stack(mask_enc_co)

            enc_ipt, enc_mask = torch.from_numpy(enc_ipt).cuda(), torch.from_numpy(enc_mask).cuda()
      
            ground = torch.from_numpy(X_batch).cuda()
   
            if model_dict['whether_prompt'] == "normal_unsup" or model_dict['whether_prompt'] == "Novel_Prompting" or model_dict['whether_prompt'] == "Normal_Prompting":
                enc_mask = copy.deepcopy(enc_ipt)
           
                prediction = net(enc_ipt * mask_enc_co, enc_mask * mask_enc_co)
         
            
                ground = ground[:, 12:, -1]
                mask_enc_co = mask_enc_co[:, 12:, -1]

            else:
                prediction = net(enc_ipt * mask_enc_co, enc_mask * mask_enc_co[:, :, :5], None, None)
            z = prediction.squeeze()
  
            loss = loss_criterion(ground, z, mask_enc_co)
      
        mae = np.mean(np.absolute(z.detach().cpu().numpy() - ground.detach().cpu().numpy()))
        if model_dict['former_pretrain'] != True:
            rmse = loss.detach().cpu().numpy()
            rmse = rmse**0.5
        else:
            loss = torch.sum(loss) / len(loss)
            rmse= rmse_cri(z, ground, mask_enc_co)
            rmse = loss.detach().cpu().numpy()**0.5

        epoch_training_mae.append(mae)
        epoch_training_rmse.append(rmse)

        loss.backward()
        optimizer.step()
        epoch_training_losses.append(loss.detach().cpu().numpy())

    epoch_metrics['loss'] = sum(epoch_training_losses)/len(epoch_training_losses)
    epoch_metrics['mae'] = sum(epoch_training_mae)/len(epoch_training_mae)
    epoch_metrics['rmse'] = sum(epoch_training_rmse)/len(epoch_training_rmse)

    return epoch_metrics


def val_fedformer_epoch(net, loss_criterion, rmse_cri, val_sequence, num_ipt, num_pre, batch_size, model_dict):

    permutation = torch.randperm(val_sequence.shape[0])
    epoch_val_losses = []
    epoch_val_mae = []
    epoch_val_rmse = []
    rmse_val = 0
    epoch_metrics = OrderedDict()

    for i in range(0, (val_sequence.shape[0] // batch_size) * batch_size, batch_size):

        indices = permutation[i:i + batch_size]
        mask_enc_co = []
        mask_benc_co = []

        X_batch = val_sequence[indices]
        # (128, 30, 12) (128, 18)
        X_batch = X_batch.squeeze()
        ground = copy.deepcopy(X_batch)[:,  num_ipt:, -1]
        
        enc_ipt = X_batch
        enc_mask = copy.copy(enc_ipt)[:, :, :5]

        if model_dict['former_pretrain'] != True:
            dec_ipt = X_batch[:, -num_pre:, :]

            dec_ipt = np.concatenate([dec_ipt, np.zeros_like(dec_ipt[:, :, :])], axis=1)
            
            # informer 2 * num_pre -> num_pre else num_pre -> 2 * num_pre
            dec_mask = copy.deepcopy(dec_ipt)[:, :, :5]
            enc_ipt, enc_mask, dec_ipt, dec_mask = torch.from_numpy(enc_ipt).cuda(), torch.from_numpy(enc_mask).cuda(), \
                                                        torch.from_numpy(dec_ipt).cuda(), torch.from_numpy(dec_mask).cuda()
            prediction = net(enc_ipt, enc_mask, dec_ipt, dec_mask)
            z = prediction.squeeze()
            ground = torch.from_numpy(ground).cuda()
            loss = loss_criterion(ground, z)
        else:
            if model_dict['whether_prompt'] == "normal_unsup" or model_dict['whether_prompt'] == "Novel_Prompting"or model_dict['whether_prompt'] == "Normal_Prompting":
                for i in range(len(enc_ipt)):
                    mask_enc = torch.ones((27, 12), dtype=bool).cuda()
                    mask_enc[:model_dict['ipt_len'], :] = torch.zeros_like(mask_enc[:model_dict['ipt_len'], :], dtype=bool)
                    mask_enc_co.append(mask_enc)
            else:
                for i in range(len(enc_ipt)):
                    mask_enc =  noise_mask(enc_ipt[i], masking_ratio=0.15, lm=3)
                    mask_enc = torch.from_numpy(mask_enc).cuda()
                    mask_enc_co.append(mask_enc)

            mask_enc_co = torch.stack(mask_enc_co)

            enc_ipt, enc_mask = torch.from_numpy(enc_ipt).cuda(), torch.from_numpy(enc_mask).cuda()
            ground = torch.from_numpy(X_batch).cuda()
            if model_dict['whether_prompt'] == "normal_unsup" or model_dict['whether_prompt'] == "Novel_Prompting" or model_dict['whether_prompt'] == "Normal_Prompting":
                enc_mask = copy.deepcopy(enc_ipt)
                prediction = net(enc_ipt * mask_enc_co, enc_mask * mask_enc_co)
                ground = ground[:, 12:, -1]
                mask_enc_co = mask_enc_co[:, 12:, -1]
            else:
                prediction = net(enc_ipt * mask_enc_co, enc_mask * mask_enc_co[:, :, :5], None, None)
            z = prediction.squeeze()

            loss = loss_criterion(ground, z, mask_enc_co)


        mae = np.mean(np.absolute(z.detach().cpu().numpy() - ground.detach().cpu().numpy()))
        if model_dict['former_pretrain'] != True:
            rmse = loss.detach().cpu().numpy()
            rmse = rmse**0.5
        else:
            loss = torch.sum(loss) / len(loss)
            rmse= rmse_cri(z, ground, mask_enc_co)
            rmse = loss.detach().cpu().numpy()**0.5

        epoch_val_mae.append(mae)
        epoch_val_rmse.append(rmse)
        epoch_val_losses.append(loss.detach().cpu().numpy())
    
    epoch_metrics['loss'] = sum(epoch_val_losses)/len(epoch_val_losses)
    epoch_metrics['mae'] = sum(epoch_val_mae)/len(epoch_val_mae)
    epoch_metrics['rmse'] = sum(epoch_val_rmse)/len(epoch_val_rmse)

    return epoch_metrics

