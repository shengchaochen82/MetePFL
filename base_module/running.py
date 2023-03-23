import logging
import sys
import os
import traceback
import json
from datetime import datetime
import string
import random
from collections import OrderedDict
import time
import pickle
from functools import partial

import ipdb
import torch
from torch.utils.data import DataLoader
import numpy as np
import sklearn

from .utils import *
from .loss import MaskedRMSE, l2_reg_loss, MaskedMAE
from .dataset import ImputationDataset, TransductionDataset, ClassiregressionDataset, collate_unsuperv, collate_superv


logger = logging.getLogger('__main__')

NEG_METRICS = {'loss'}  # metrics for which "better" is less

val_times = {"total_time": 0, "count": 0}


def setup(args):
    """Prepare training session: read configuration from file (takes precedence), create directories.
    Input:
        args: arguments object from argparse
    Returns:
        config: configuration dictionary
    """

    config = args.__dict__  # configuration dictionary
    random.seed(1334)
    os.environ['PYTHONHASHSEED'] = str(1334)
    np.random.seed(1334)
    torch.manual_seed(1334)
    torch.cuda.manual_seed_all(1334)
    if args.config_filepath is not None:
        logger.info("Reading configuration ...")
        try:  # dictionary containing the entire configuration settings in a hierarchical fashion
            config.update(load_config(args.config_filepath))
        except:
            logger.critical("Failed to load configuration file. Check JSON syntax and verify that files exist")
            traceback.print_exc()
            sys.exit(1)

    # Create output directory
    initial_timestamp = datetime.now()
    output_dir = config['output_dir']
    if not os.path.isdir(output_dir):
        raise IOError(
            "Root directory '{}', where the directory of the experiment will be created, must exist".format(output_dir))

    output_dir = os.path.join(output_dir, config['experiment_name'])

    formatted_timestamp = initial_timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    config['initial_timestamp'] = formatted_timestamp
    if (not config['no_timestamp']) or (len(config['experiment_name']) == 0):
        rand_suffix = "".join(random.choices(string.ascii_letters + string.digits, k=3))
        output_dir += "_" + formatted_timestamp + "_" + rand_suffix
    config['output_dir'] = output_dir
    config['save_dir'] = os.path.join(output_dir, 'checkpoints')
    config['pred_dir'] = os.path.join(output_dir, 'predictions')
    config['tensorboard_dir'] = os.path.join(output_dir, 'tb_summaries')
    create_dirs([config['save_dir'], config['pred_dir'], config['tensorboard_dir']])

    # Save configuration as a (pretty) json file
    with open(os.path.join(output_dir, 'configuration.json'), 'w') as fp:
        json.dump(config, fp, indent=4, sort_keys=True)

    logger.info("Stored configuration file in '{}'".format(output_dir))

    return config


def fold_evaluate(dataset, model, device, loss_module, target_feats, config, dataset_name):

    allfolds = {'target_feats': target_feats,  # list of len(num_folds), each element: list of target feature integer indices
                'predictions': [],  # list of len(num_folds), each element: (num_samples, seq_len, feat_dim) prediction per sample
                'targets': [],  # list of len(num_folds), each element: (num_samples, seq_len, feat_dim) target/original input per sample
                'target_masks': [],  # list of len(num_folds), each element: (num_samples, seq_len, feat_dim) boolean mask per sample
                'metrics': [],  # list of len(num_folds), each element: (num_samples, num_metrics) metric per sample
                'IDs': []}  # list of len(num_folds), each element: (num_samples,) ID per sample

    for i, tgt_feats in enumerate(target_feats):

        dataset.mask_feats = tgt_feats  # set the transduction target features

        loader = DataLoader(dataset=dataset,
                            batch_size=config['batch_size'],
                            shuffle=False,
                            num_workers=config['num_workers'],
                            pin_memory=True,
                            collate_fn=lambda x: collate_unsuperv(x, max_len=config['max_seq_len']))

        evaluator = UnsupervisedRunner(model, loader, device, loss_module,
                                       print_interval=config['print_interval'], console=config['console'])

        logger.info("Evaluating {} set, fold: {}, target features: {}".format(dataset_name, i, tgt_feats))
        aggr_metrics, per_batch = evaluate(evaluator)

        metrics_array = convert_metrics_per_batch_to_per_sample(per_batch['metrics'], per_batch['target_masks'])
        metrics_array = np.concatenate(metrics_array, axis=0)
        allfolds['metrics'].append(metrics_array)
        allfolds['predictions'].append(np.concatenate(per_batch['predictions'], axis=0))
        allfolds['targets'].append(np.concatenate(per_batch['targets'], axis=0))
        allfolds['target_masks'].append(np.concatenate(per_batch['target_masks'], axis=0))
        allfolds['IDs'].append(np.concatenate(per_batch['IDs'], axis=0))

        metrics_mean = np.mean(metrics_array, axis=0)
        metrics_std = np.std(metrics_array, axis=0)
        for m, metric_name in enumerate(list(aggr_metrics.items())[1:]):
            logger.info("{}:: Mean: {:.3f}, std: {:.3f}".format(metric_name, metrics_mean[m], metrics_std[m]))

    pred_filepath = os.path.join(config['pred_dir'], dataset_name + '_fold_transduction_predictions.pickle')
    logger.info("Serializing predictions into {} ... ".format(pred_filepath))
    with open(pred_filepath, 'wb') as f:
        pickle.dump(allfolds, f, pickle.HIGHEST_PROTOCOL)


def convert_metrics_per_batch_to_per_sample(metrics, target_masks):
    """
    Args:
        metrics: list of len(num_batches), each element: list of len(num_metrics), each element: (num_active_in_batch,) metric per element
        target_masks: list of len(num_batches), each element: (batch_size, seq_len, feat_dim) boolean mask: 1s active, 0s ignore
    Returns:
        metrics_array = list of len(num_batches), each element: (batch_size, num_metrics) metric per sample
    """
    metrics_array = []
    for b, batch_target_masks in enumerate(target_masks):
        num_active_per_sample = np.sum(batch_target_masks, axis=(1, 2))
        batch_metrics = np.stack(metrics[b], axis=1)  # (num_active_in_batch, num_metrics)
        ind = 0
        metrics_per_sample = np.zeros((len(num_active_per_sample), batch_metrics.shape[1]))  # (batch_size, num_metrics)
        for n, num_active in enumerate(num_active_per_sample):
            new_ind = ind + num_active
            metrics_per_sample[n, :] = np.sum(batch_metrics[ind:new_ind, :], axis=0)
            ind = new_ind
        metrics_array.append(metrics_per_sample)
    return metrics_array


def evaluate(evaluator):
    """Perform a single, one-off evaluation on an evaluator object (initialized with a dataset)"""
    eval_start_time = time.time()
    with torch.no_grad():
        aggr_metrics, per_batch = evaluator.evaluate(epoch_num=None, keep_all=True)
    eval_runtime = time.time() - eval_start_time
    print()
    print_str = 'Evaluation Summary: '
    for k, v in aggr_metrics.items():
        if v is not None:
            print_str += '{}: {:8f} | '.format(k, v)
    logger.info(print_str)
    logger.info("Evaluation runtime: {} hours, {} minutes, {} seconds\n".format(*readable_time(eval_runtime)))

    return aggr_metrics, per_batch


def validate(val_evaluator, tensorboard_writer, config, best_metrics, best_value, epoch):
    """Run an evaluation on the validation set while logging metrics, and handle outcome"""

    logger.info("Evaluating on validation set ...")
    eval_start_time = time.time()
    with torch.no_grad():
        aggr_metrics, per_batch = val_evaluator.evaluate(epoch, keep_all=True)
    eval_runtime = time.time() - eval_start_time
    logger.info("Validation runtime: {} hours, {} minutes, {} seconds\n".format(*readable_time(eval_runtime)))

    global val_times
    val_times["total_time"] += eval_runtime
    val_times["count"] += 1
    avg_val_time = val_times["total_time"] / val_times["count"]
    avg_val_batch_time = avg_val_time / len(val_evaluator.dataloader)
    avg_val_sample_time = avg_val_time / len(val_evaluator.dataloader.dataset)
    logger.info("Avg val. time: {} hours, {} minutes, {} seconds".format(*readable_time(avg_val_time)))
    logger.info("Avg batch val. time: {} seconds".format(avg_val_batch_time))
    logger.info("Avg sample val. time: {} seconds".format(avg_val_sample_time))

    print()
    print_str = 'Epoch {} Validation Summary: '.format(epoch)
    for k, v in aggr_metrics.items():
        tensorboard_writer.add_scalar('{}/val'.format(k), v, epoch)
        print_str += '{}: {:8f} | '.format(k, v)
    logger.info(print_str)

    if config['key_metric'] in NEG_METRICS:
        condition = (aggr_metrics[config['key_metric']] < best_value)
    else:
        condition = (aggr_metrics[config['key_metric']] > best_value)
    if condition:
        best_value = aggr_metrics[config['key_metric']]
        save_model(os.path.join(config['save_dir'], 'model_best.pth'), epoch, val_evaluator.model)
        best_metrics = aggr_metrics.copy()

        pred_filepath = os.path.join(config['pred_dir'], 'best_predictions')
        np.savez(pred_filepath, **per_batch)

    return aggr_metrics, best_metrics, best_value



def check_progress(epoch):

    if epoch in [100, 140, 160, 220, 280, 340]:
        return True
    else:
        return False


class BaseRunner(object):

    def __init__(self, model, dataloader, device, loss_module, optimizer=None, l2_reg=None, print_interval=10, console=True, model_dict=None, global_param=None, local_param=None, server_param=None):

        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.optimizer = optimizer
        self.loss_module = loss_module
        self.l2_reg = l2_reg
        self.print_interval = print_interval
        self.model_dict = model_dict
        self.printer = Printer(console=console)
        self.mae = MaskedMAE('mean')
        self.rmse = MaskedRMSE('mean')
        self.epoch_metrics = OrderedDict()
        self.global_param = global_param
        self.local_param = local_param
        self.server_param = server_param

    def train_epoch(self, epoch_num=None):
        raise NotImplementedError('Please override in child class')

    def evaluate(self, epoch_num=None, keep_all=False):
        raise NotImplementedError('Please override in child class')

    def print_callback(self, i_batch, metrics, prefix=''):

        total_batches = len(self.dataloader)

        template = "{:5.1f}% | batch: {:9d} of {:9d}"
        content = [100 * (i_batch / total_batches), i_batch, total_batches]
        for met_name, met_value in metrics.items():
            template += "\t|\t{}".format(met_name) + ": {:g}"
            content.append(met_value)

        dyn_string = template.format(*content)
        dyn_string = prefix + dyn_string
        self.printer.print(dyn_string)


class UnsupervisedRunner(BaseRunner):
    """
    This UnsupervisedRunner both for Unsupervised pretrain and supervised federated prompt learning, change only computational mean of loss
    """
    def train_epoch(self, epoch_num=None):

        self.model = self.model.train()

        epoch_loss = 0  # total loss of epoch
        epoch_mae = 0
        epoch_rmse = 0
        total_active_elements = 0  # total unmasked elements in epoch
        for i, batch in enumerate(self.dataloader):

            X, targets, target_masks, padding_masks, IDs = batch
            
            targets = targets.to(self.device)
            target_masks = target_masks.to(self.device)  # 1s: mask and predict, 0s: unaffected input (ignore)
            padding_masks = padding_masks.to(self.device)  # 0s: ignore

            predictions = self.model(X.to(self.device), padding_masks)  # (batch_size, padded_length, feat_dim)

            # Cascade noise masks (batch_size, padded_length, feat_dim) and padding masks (batch_size, padded_length)
            target_masks = target_masks * padding_masks.unsqueeze(-1)

            """
                Federated Prompting Learning Loss calculation
            """
            if self.model_dict['pre_train'] == True:
                loss = self.loss_module(predictions, targets, target_masks)
                mae = self.mae(predictions, targets, target_masks)
                rmse = self.rmse(predictions, targets, target_masks)
            else:
                if self.model_dict['whether_prompt'] == "normal_unsup" or self.model_dict['whether_prompt'] == "Novel_Prompting" or self.model_dict['whether_prompt'] == "Normal_Prompting":
                    pk = predictions.detach().cpu().numpy()
                    pk = np.nan_to_num(pk)
                    pk[np.isinf(pk)] = 0.
                    predictions = torch.from_numpy(pk).cuda()
                    # print(predictions.shape, targets[:, self.model_dict['ipt_len'] : self.model_dict['fore_len'] + self.model_dict['pro_len'], -1].shape, target_masks[:, self.model_dict['ipt_len'] : self.model_dict['fore_len'] + self.model_dict['pro_len'], -1].shape)
                    loss = self.loss_module(predictions, 
                                            targets[:, self.model_dict['ipt_len'] : self.model_dict['fore_len'] + self.model_dict['pro_len'], -1], 
                                            target_masks[:, self.model_dict['ipt_len'] : self.model_dict['fore_len'] + self.model_dict['pro_len'], -1])

                    mae = self.mae(predictions, 
                                            targets[:, self.model_dict['ipt_len'] : self.model_dict['fore_len'] + self.model_dict['pro_len'], -1], 
                                            target_masks[:, self.model_dict['ipt_len'] : self.model_dict['fore_len'] + self.model_dict['pro_len'], -1])

                    rmse = self.rmse(predictions, 
                                                targets[:, self.model_dict['ipt_len'] : self.model_dict['fore_len'] + self.model_dict['pro_len'], -1], 
                                                target_masks[:, self.model_dict['ipt_len'] : self.model_dict['fore_len'] + self.model_dict['pro_len'], -1])
                else:

                    loss = self.loss_module(predictions[:, self.model_dict['ipt_len'] : self.model_dict['fore_len'] + self.model_dict['pro_len'], -1], 
                                            targets[:, self.model_dict['ipt_len'] : self.model_dict['fore_len'] + self.model_dict['pro_len'], -1], 
                                            target_masks[:, self.model_dict['ipt_len'] : self.model_dict['fore_len'] + self.model_dict['pro_len'], -1])

                    mae = self.mae(predictions[:, self.model_dict['ipt_len'] : self.model_dict['fore_len'] + self.model_dict['pro_len'], -1], 
                                            targets[:, self.model_dict['ipt_len'] : self.model_dict['fore_len'] + self.model_dict['pro_len'], -1], 
                                            target_masks[:, self.model_dict['ipt_len'] : self.model_dict['fore_len'] + self.model_dict['pro_len'], -1])

                    rmse = self.rmse(predictions[:, self.model_dict['ipt_len'] : self.model_dict['fore_len'] + self.model_dict['pro_len'], -1], 
                                                targets[:, self.model_dict['ipt_len'] : self.model_dict['fore_len'] + self.model_dict['pro_len'], -1], 
                                                target_masks[:, self.model_dict['ipt_len'] : self.model_dict['fore_len'] + self.model_dict['pro_len'], -1])
                


            batch_loss = torch.sum(loss)

            batch_rmse = np.sum(rmse)
            mean_loss = batch_loss / len(loss)  # mean loss (over active elements) used for optimization
    
            # Reg-Prompt
            if self.l2_reg:
                total_loss = mean_loss + self.l2_reg * self.local_criterion(self.model)
            else:
                total_loss = mean_loss

            # Zero gradients, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()
            # loss.requires_grad_(True)   
            total_loss.requires_grad_(True)
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
            self.optimizer.step()

            metrics = {"loss": mean_loss.item()}

            with torch.no_grad():
                total_active_elements += len(loss)
                epoch_loss += batch_loss.item()  # add total loss of batch
                epoch_mae += mae.item()
                epoch_rmse += batch_rmse.item()

        epoch_loss = epoch_loss / total_active_elements  # average loss per element for whole epoch
        epoch_mae = epoch_mae / total_active_elements 
        epoch_rmse = epoch_rmse / len(self.dataloader)

        self.epoch_metrics['epoch'] = epoch_num
        self.epoch_metrics['loss'] = epoch_loss
        self.epoch_metrics['mae'] = epoch_mae
        self.epoch_metrics['rmse'] = epoch_rmse
        return self.epoch_metrics
        
    def local_criterion(self, loss):

        self.m1 = sd_matrixing_prompt(self.model.get_state()[0]).reshape(1, -1).cuda()
        # Actually is Zero 
        self.m2 = sd_matrixing_prompt(self.server_param).reshape(1, -1).cuda()
        self.m3 = sd_matrixing_prompt(self.global_param).reshape(1, -1).cuda()
        self.reg1 = torch.nn.functional.pairwise_distance(self.m1, self.m2, p=2)
        self.reg2 = torch.nn.functional.pairwise_distance(self.m1, self.m3, p=2)
        
        return loss

    def evaluate(self, epoch_num=None, keep_all=False):

        self.model = self.model.eval()

        epoch_loss = 0  # total loss of epoch
        epoch_mae = 0
        epoch_rmse = 0
        total_active_elements = 0  # total unmasked elements in epoch

        if keep_all:
            per_batch = {'target_masks': [], 'targets': [], 'predictions': [], 'metrics': [], 'IDs': []}
        for i, batch in enumerate(self.dataloader):
            
            X, targets, target_masks, padding_masks, IDs = batch
            targets = targets.to(self.device)
            target_masks = target_masks.to(self.device)  # 1s: mask and predict, 0s: unaffected input (ignore)
            padding_masks = padding_masks.to(self.device)  # 0s: ignore
            predictions = self.model(X.to(self.device), padding_masks)  # (batch_size, padded_length, feat_dim)
            # Cascade noise masks (batch_size, padded_length, feat_dim) and padding masks (batch_size, padded_length)
            target_masks = target_masks * padding_masks.unsqueeze(-1)


            """
                Federated Prompting Learning Loss calculation
            """
            if self.model_dict['pre_train'] == True:
                loss = self.loss_module(predictions, targets, target_masks)
                mae = self.mae(predictions, targets, target_masks)
                rmse = self.rmse(predictions, targets, target_masks)
            else:
                if self.model_dict['whether_prompt'] == "normal_unsup" or self.model_dict['whether_prompt'] == "Novel_Prompting" or self.model_dict['whether_prompt'] == "Normal_Prompting":
                    pk = predictions.detach().cpu().numpy()
                    pk = np.nan_to_num(pk)
                    pk[np.isinf(pk)] = 0.
                    predictions = torch.from_numpy(pk).cuda()
                    loss = self.loss_module(predictions, 
                                            targets[:, self.model_dict['ipt_len'] : self.model_dict['fore_len'] + self.model_dict['pro_len'], -1], 
                                            target_masks[:, self.model_dict['ipt_len'] : self.model_dict['fore_len'] + self.model_dict['pro_len'], -1])

                    mae = self.mae(predictions, 
                                            targets[:, self.model_dict['ipt_len'] : self.model_dict['fore_len'] + self.model_dict['pro_len'], -1], 
                                            target_masks[:, self.model_dict['ipt_len'] : self.model_dict['fore_len'] + self.model_dict['pro_len'], -1])

                    rmse = self.rmse(predictions, 
                                                targets[:, self.model_dict['ipt_len'] : self.model_dict['fore_len'] + self.model_dict['pro_len'], -1], 
                                                target_masks[:, self.model_dict['ipt_len'] : self.model_dict['fore_len'] + self.model_dict['pro_len'], -1])
                else:

                    loss = self.loss_module(predictions[:, self.model_dict['ipt_len'] : self.model_dict['fore_len'] + self.model_dict['pro_len'], -1], 
                                            targets[:, self.model_dict['ipt_len'] : self.model_dict['fore_len'] + self.model_dict['pro_len'], -1], 
                                            target_masks[:, self.model_dict['ipt_len'] : self.model_dict['fore_len'] + self.model_dict['pro_len'], -1])

                    mae = self.mae(predictions[:, self.model_dict['ipt_len'] : self.model_dict['fore_len'] + self.model_dict['pro_len'], -1], 
                                            targets[:, self.model_dict['ipt_len'] : self.model_dict['fore_len'] + self.model_dict['pro_len'], -1], 
                                            target_masks[:, self.model_dict['ipt_len'] : self.model_dict['fore_len'] + self.model_dict['pro_len'], -1])

                    rmse = self.rmse(predictions[:, self.model_dict['ipt_len'] : self.model_dict['fore_len'] + self.model_dict['pro_len'], -1], 
                                                targets[:, self.model_dict['ipt_len'] : self.model_dict['fore_len'] + self.model_dict['pro_len'], -1], 
                                                target_masks[:, self.model_dict['ipt_len'] : self.model_dict['fore_len'] + self.model_dict['pro_len'], -1])
            
            batch_loss = torch.sum(loss)
            batch_rmse = np.sum(rmse)
            mean_loss = batch_loss / len(loss)  # mean loss (over active elements) used for optimization the batch
            
            """
            Unmasked on the process of pre-trained, these need unmasked when prompting learning activation
            """
            if self.model_dict['pre_train'] == True:
                if keep_all:
                    per_batch['target_masks'].append(target_masks.cpu().numpy())
                    per_batch['targets'].append(targets.cpu().numpy())
                    per_batch['predictions'].append(predictions.cpu().numpy())
                    per_batch['metrics'].append([loss.cpu().numpy()])
                    per_batch['IDs'].append(IDs)

            metrics = {"loss": mean_loss}
            # if i % self.print_interval == 0:
            #     ending = "" if epoch_num is None else 'Epoch {} '.format(epoch_num)
            #     self.print_callback(i, metrics, prefix='Evaluating ' + ending)

            total_active_elements += len(loss)
            epoch_mae += mae.item()
            epoch_loss += batch_loss.item()  # add total loss of batch
            epoch_rmse += batch_rmse.item()

        epoch_mae = epoch_mae / total_active_elements 
        epoch_loss = epoch_loss / total_active_elements
        epoch_rmse = epoch_rmse / len(self.dataloader)

        self.epoch_metrics['epoch'] = epoch_num
        self.epoch_metrics['loss'] = epoch_loss
        self.epoch_metrics['mae'] = epoch_mae
        self.epoch_metrics['rmse'] = epoch_rmse


        if keep_all:
            return self.epoch_metrics, per_batch
        else:
            return self.epoch_metrics

def sd_matrixing_prompt(state_dic):
    """
    Turn state dic into a vector
    :param state_dic:
    :return:
    """
    keys = []
    param_vector = None
    for key, param in state_dic.items():
        # print(key)
        col = key.split('_')
        if col != "Transformer_prompt_pre":
            keys.append(key)
            if param_vector is None:
                param_vector = param.clone().detach().flatten().cpu()
            else:
                if len(list(param.size())) == 0:
                    param_vector = torch.cat((param_vector, param.clone().detach().view(1).cpu().type(torch.float32)), 0)
                else:

                    param_vector = torch.cat((param_vector, param.clone().detach().flatten().cpu()), 0)
    return param_vector