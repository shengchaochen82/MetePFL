import datetime
import logging
import random
import os
import torch
import numpy as np
from collections import namedtuple
from functools import singledispatch
from base_module.dataset import collate_unsuperv
import xlrd
import xlwt
from xlutils.copy import copy
from base_module.promp_mask import Imputation_Prompting_Dataset
from base_module.running import UnsupervisedRunner
from torch.utils.data import DataLoader
import logging
logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def init_model(dataset, data_indices, dict_user, client_dict):
    train_dataset = Imputation_Prompting_Dataset(dataset[:, list(dict_user), :, :], data_indices, 0.2, 3, 12, 21, 3)
    train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=client_dict['batch_size'],
            shuffle=False,
            num_workers=client_dict['num_workers'],
            pin_memory=True,
            collate_fn=lambda x : collate_unsuperv(x, max_len=24)
        )
    return train_loader


def print2file(buf, out_file, p=False):
    if p:
        print(buf)
    outfd = open(out_file, 'a+')
    outfd.write(str(datetime.datetime.now()) + '\t' + buf + '\n')
    outfd.close()


def initial_environment(seed, cpu_num=5, deterministic=False):
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def sd_matrixing(state_dic):
    """
    Turn state dic into a vector
    :param state_dic:
    :return:
    """
    keys = []
    param_vector = None
    for key, param in state_dic.items():
        keys.append(key)
        if param_vector is None:
            param_vector = param.clone().detach().flatten().cpu()
        else:
            if len(list(param.size())) == 0:
                param_vector = torch.cat((param_vector, param.clone().detach().view(1).cpu().type(torch.float32)), 0)
            else:

                param_vector = torch.cat((param_vector, param.clone().detach().flatten().cpu()), 0)
    return param_vector

def state_decom(state_dic, model_name):
    """
    Turn state dic into a vector
    :param state_dic:
    :return:
    """
    keys = []
    param_vector = None
    for key, param in state_dic.items():
        col = key.split('.')[0]
        if col != model_name:
            keys.append(key)
            if param_vector is None:
                param_vector = param.clone().detach().flatten().cpu()
            else:
                if len(list(param.size())) == 0:
                    param_vector = torch.cat((param_vector, param.clone().detach().view(1).cpu().type(torch.float32)), 0)
                else:
                    param_vector = torch.cat((param_vector, param.clone().detach().flatten().cpu()), 0)

    return param_vector




def trainable_params(model):
    result = []
    for p in model.parameters():
        if p.requires_grad:
            result.append(p)
    return result


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger



class PiecewiseLinear(namedtuple('PiecewiseLinear', ('knots', 'vals'))):
    def __call__(self, t):
        return np.interp([t], self.knots, self.vals)[0]


class StatsLogger():
    def __init__(self, keys):
        self._stats = {k: [] for k in keys}

    def append(self, output):
        for k, v in self._stats.items():
            v.append(output[k].detach())

    def stats(self, key):
        return cat(*self._stats[key])

    def mean(self, key):
        return np.mean(to_numpy(self.stats(key)), dtype=np.float)


@singledispatch
def cat(*xs):
    raise NotImplementedError


@singledispatch
def to_numpy(x):
    raise NotImplementedError


@cat.register(torch.Tensor)
def _(*xs):
    return torch.cat(xs)


@to_numpy.register(torch.Tensor)
def _(x):
    return x.detach().cpu().numpy()


def export_performance_metrics(filepath, metrics_table, header, book=None, sheet_name="metrics"):
    """Exports performance metrics on the validation set for all epochs to an excel file"""

    if book is None:
        book = xlwt.Workbook()  # new excel work book

    book = write_table_to_sheet([header] + metrics_table, book, sheet_name=sheet_name)

    book.save(filepath)
    logger.info("Exported per epoch performance metrics in '{}'".format(filepath))

    return book


def write_row(sheet, row_ind, data_list):
    """Write a list to row_ind row of an excel sheet"""

    row = sheet.row(row_ind)
    for col_ind, col_value in enumerate(data_list):
        row.write(col_ind, col_value)
    return


def write_table_to_sheet(table, work_book, sheet_name=None):
    """Writes a table implemented as a list of lists to an excel sheet in the given work book object"""

    sheet = work_book.add_sheet(sheet_name)

    for row_ind, row_list in enumerate(table):
        write_row(sheet, row_ind, row_list)

    return work_book


def export_record(filepath, values):
    """Adds a list of values as a bottom row of a table in a given excel file"""

    read_book = xlrd.open_workbook(filepath, formatting_info=True)
    read_sheet = read_book.sheet_by_index(0)
    last_row = read_sheet.nrows

    work_book = copy(read_book)
    sheet = work_book.get_sheet(0)
    write_row(sheet, last_row, values)
    work_book.save(filepath)


def register_record(filepath, timestamp, experiment_name, best_metrics, final_metrics=None, comment=''):
    """
    Adds the best and final metrics of a given experiment as a record in an excel sheet with other experiment records.
    Creates excel sheet if it doesn't exist.
    Args:
        filepath: path of excel file keeping records
        timestamp: string
        experiment_name: string
        best_metrics: dict of metrics at best epoch {metric_name: metric_value}. Includes "epoch" as first key
        final_metrics: dict of metrics at final epoch {metric_name: metric_value}. Includes "epoch" as first key
        comment: optional description
    """
    metrics_names, metrics_values = zip(*best_metrics.items())
    row_values = [timestamp, experiment_name, comment] + list(metrics_values)
    if final_metrics is not None:
        final_metrics_names, final_metrics_values = zip(*final_metrics.items())
        row_values += list(final_metrics_values)

    if not os.path.exists(filepath):  # Create a records file for the first time
        logger.warning("Records file '{}' does not exist! Creating new file ...".format(filepath))
        directory = os.path.dirname(filepath)
        if len(directory) and not os.path.exists(directory):
            os.makedirs(directory)
        header = ["Timestamp", "Name", "Comment"] + ["Best " + m for m in metrics_names]
        if final_metrics is not None:
            header += ["Final " + m for m in final_metrics_names]
        book = xlwt.Workbook()  # excel work book
        book = write_table_to_sheet([header, row_values], book, sheet_name="records")
        book.save(filepath)
    else:
        try:
            export_record(filepath, row_values)
        except Exception as x:
            alt_path = os.path.join(os.path.dirname(filepath), "record_" + experiment_name)
            logger.error("Failed saving in: '{}'! Will save here instead: {}".format(filepath, alt_path))
            export_record(alt_path, row_values)
            filepath = alt_path

    logger.info("Exported performance record to '{}'".format(filepath))