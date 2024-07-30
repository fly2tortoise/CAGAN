
import os
import torch
import dateutil.tz
from datetime import datetime
import time
import logging

import numpy as np


def create_logger(log_dir, phase='train'):
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}.log'.format(time_str, phase)
    final_log_file = os.path.join(log_dir, log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger


def set_log_dir(root_dir, exp_name):
    path_dict = {}
    os.makedirs(root_dir, exist_ok=True)

    # set log path
    exp_path = os.path.join(root_dir, exp_name)
    now = datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    prefix = exp_path + '_' + timestamp
    os.makedirs(prefix)
    path_dict['prefix'] = prefix

    # set checkpoint path
    ckpt_path = os.path.join(prefix, 'Model')
    os.makedirs(ckpt_path)
    path_dict['ckpt_path'] = ckpt_path

    log_path = os.path.join(prefix, 'Log')
    os.makedirs(log_path)
    path_dict['log_path'] = log_path

    # set sample image path for fid calculation
    sample_path = os.path.join(prefix, 'Samples')
    os.makedirs(sample_path)
    path_dict['sample_path'] = sample_path

    genotypes_path = os.path.join(prefix, 'Genotypes')
    os.makedirs(genotypes_path)
    path_dict['genotypes_path'] = genotypes_path

    graph_vis_path = os.path.join(prefix, 'Graph_vis')
    os.makedirs(graph_vis_path)
    path_dict['graph_vis_path'] = graph_vis_path

    return path_dict


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth'):
    epoch = states['epoch']
    if epoch>=300:
        filename = str(epoch) + 'checkpoint_best.pth'
    torch.save(states, os.path.join(output_dir, filename))
    if is_best:
        torch.save(states, os.path.join(output_dir, 'checkpoint_best.pth'))

def save_is_checkpoint(states, is_best, output_dir):
    if is_best:
        torch.save(states, os.path.join(output_dir, 'checkpoint_best_is.pth'))

def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6

def count_parameters_in_Arch_shape_MB(model):
    # 3 cell\ 6 cell-shape\ each-cell-shape
    sum_cell1, sum_cell2, sum_cell3 = 0, 0, 0
    sum_cell1_1, sum_cell1_2, sum_cell2_1, sum_cell2_2, sum_cell3_1, sum_cell3_2 = 0, 0, 0, 0, 0, 0
    for name, v in model.named_parameters():
        print(name)
        if "auxiliary" not in name:
            if "module.cell1" in name:
                sum_cell1 += np.prod(v.size())
            if "module.cell2" in name:
                sum_cell2 += np.prod(v.size())
            if "module.cell3" in name:
                sum_cell3 += np.prod(v.size())

            if "module.cell1" in name and ("module.cell1.c0" in name or "module.cell1.c1" in name or "module.cell1.up" in name):
                sum_cell1_1 += np.prod(v.size())
            if "module.cell1" in name and ("module.cell1.c2" in name or "module.cell1.c3" in name or "module.cell1.c4" in name):
                sum_cell1_2 += np.prod(v.size())
            if "module.cell2" in name and ("module.cell2.c0" in name or "module.cell2.c1" in name or "module.cell2.up" in name):
                sum_cell2_1 += np.prod(v.size())
            if "module.cell2" in name and ("module.cell2.c2" in name or "module.cell2.c3" in name or "module.cell2.c4" in name):
                sum_cell2_2 += np.prod(v.size())
            if "module.cell3" in name and ("module.cell3.c0" in name or "module.cell3.c1" in name or "module.cell3.up" in name):
                sum_cell3_1 += np.prod(v.size())
            if "module.cell3" in name and ("module.cell3.c2" in name or "module.cell3.c3" in name or "module.cell3.c4" in name):
                sum_cell3_2 += np.prod(v.size())

    # 3 layer
    cell_3_layer = []
    sum_cell1 = sum_cell1 / 1e6
    sum_cell2 = sum_cell2 / 1e6
    sum_cell3 = sum_cell3 / 1e6
    cell_3_layer.extend([sum_cell1, sum_cell2, sum_cell3])

    # 6 layer
    cell_6_layer = []
    sum_cell1_1 =sum_cell1_1 / 1e6
    sum_cell1_2 =sum_cell1_2 / 1e6
    sum_cell2_1 =sum_cell2_1 / 1e6
    sum_cell2_2 =sum_cell2_2 / 1e6
    sum_cell3_1 =sum_cell3_1 / 1e6
    sum_cell3_2 =sum_cell3_2 / 1e6
    cell_6_layer.extend([sum_cell1_1, sum_cell1_2, sum_cell2_1, sum_cell2_2, sum_cell3_1, sum_cell3_2])

    return  cell_3_layer, cell_6_layer

def count_parameters_search_Arch_shape_MB(model):
    # 3 cell\ 6 cell-shape\ each-cell-shape
    sum_cell1, sum_cell2, sum_cell3 = 0, 0, 0
    sum_cell1_1, sum_cell1_2, sum_cell2_1, sum_cell2_2, sum_cell3_1, sum_cell3_2 = 0, 0, 0, 0, 0, 0
    for name, v in model.named_parameters():
        # print(name)
        if "auxiliary" not in name:
            if "cell1" in name:
                sum_cell1 += np.prod(v.size())
            if "cell2" in name:
                sum_cell2 += np.prod(v.size())
            if "cell3" in name:
                sum_cell3 += np.prod(v.size())

            if "cell1" in name and (
                    "cell1.c0" in name or "cell1.c1" in name or "cell1.up" in name):
                sum_cell1_1 += np.prod(v.size())
            if "cell1" in name and (
                    "cell1.c2" in name or "cell1.c3" in name or "cell1.c4" in name):
                sum_cell1_2 += np.prod(v.size())
            if "cell2" in name and (
                    "cell2.c0" in name or "cell2.c1" in name or "cell2.up" in name):
                sum_cell2_1 += np.prod(v.size())
            if "cell2" in name and (
                    "cell2.c2" in name or "cell2.c3" in name or "cell2.c4" in name):
                sum_cell2_2 += np.prod(v.size())
            if "cell3" in name and (
                    "cell3.c0" in name or "cell3.c1" in name or "cell3.up" in name):
                sum_cell3_1 += np.prod(v.size())
            if "cell3" in name and (
                    "cell3.c2" in name or "cell3.c3" in name or "cell3.c4" in name):
                sum_cell3_2 += np.prod(v.size())

    # 3 layer
    cell_3_layer = []
    # cell_3_layer.append(8.8)
    sum_cell1 = sum_cell1 / 1e6
    sum_cell2 = sum_cell2 / 1e6
    sum_cell3 = sum_cell3 / 1e6
    cell_3_layer.extend([sum_cell1, sum_cell2, sum_cell3])

    # 6 layer
    cell_6_layer = []
    sum_cell1_1 = sum_cell1_1 / 1e6
    sum_cell1_2 = sum_cell1_2 / 1e6
    sum_cell2_1 = sum_cell2_1 / 1e6
    sum_cell2_2 = sum_cell2_2 / 1e6
    sum_cell3_1 = sum_cell3_1 / 1e6
    sum_cell3_2 = sum_cell3_2 / 1e6
    cell_6_layer.extend([sum_cell1_1, sum_cell1_2, sum_cell2_1, sum_cell2_2, sum_cell3_1, sum_cell3_2])

    # extend 3 6 layer
    # cell_3_layer.extend(cell_6_layer)

    return cell_3_layer, cell_6_layer

def count_cosine_similarity_6layer(FID_arch_Params):
    A = np.array([ 0.8, 0.8, 0.8, 1.0, 0.7, 1.1])  # 0到6范围内的值
    scaled_A = np.array([(x / 6) * 100 for x in A])  # 缩放到0到100范围内
    B = FID_arch_Params
    scaled_B = np.array([(x / 6) * 100 for x in B])  # 缩放到0到100范围内
    # 计算余弦相似度
    dot_product = np.dot(scaled_A, scaled_B)
    norm_A = np.linalg.norm(scaled_A)
    norm_B = np.linalg.norm(scaled_B)
    similarity = dot_product / (norm_A * norm_B)
    # print(f"The cosine similarity between A and B is: {similarity}")
    return similarity

def count_cosine_similarity_3cell(FID_arch_Params):
    A = np.array([1.5, 1.8, 1.8])  # 0到6范围内的值
    scaled_A = np.array([(x / 6) * 100 for x in A])  # 缩放到0到100范围内
    B = FID_arch_Params
    scaled_B = np.array([(x / 6) * 100 for x in B])  # 缩放到0到100范围内
    # 计算余弦相似度
    dot_product = np.dot(scaled_A, scaled_B)
    norm_A = np.linalg.norm(scaled_A)
    norm_B = np.linalg.norm(scaled_B)
    similarity = dot_product / (norm_A * norm_B)
    # print(f"The cosine similarity between A and B is: {similarity}")
    return similarity

def calculate_similarity_score(reference_value_min, unknown_value, reference_value_max):
    if reference_value_min == unknown_value:
        return 1
    elif reference_value_max > unknown_value and unknown_value >= reference_value_min:
        max_score = 1
        deduction_factor = 0.05  # 每单位差距扣分数
        difference = unknown_value - reference_value_min
        similarity_score = max_score - deduction_factor * difference
        return max(0, similarity_score)  # 分数不低于0

    elif reference_value_max <= unknown_value:
        max_score = 1
        deduction_factor = 0.05  # 每单位差距扣分数
        difference = unknown_value - reference_value_min
        # print(difference)
        similarity_score = max_score - deduction_factor * difference
        return max(0, similarity_score)  # 分数不低于0
    else:
        return 1  # 未知数字比参考值小，相似性得分为100



def record(filepath=None, Arch=None,
           best_IS=None, best_IS_epoch=None,
           best_fid=None, best_fid_epoch=None):
    with open(os.path.join(filepath, 'search_line.txt'), 'a') as f:
        f.write('Arch_' + Arch+'_epoch' + ';'+
                'best_IS '+ str(best_IS) + ',' + 'at_' + best_IS_epoch + '_epoch' + ';' +
                'best_fid ' + str(best_fid) + ',' + 'at_' + best_fid_epoch + '_epoch' + '\n')

# draw the search line efficiently
def early_stop(epoch, best_IS, best_fid):
    if epoch == 20:
        if (best_IS < 1.5) and (best_fid > 250):
            return True
    if epoch == 40:
        if (best_IS < 4.7) or (best_fid > 80):
            return True
    if epoch == 60:
        if (best_IS < 7) or (best_fid > 40):
            return True
    if epoch == 80:
        if (best_IS < 7.5) or (best_fid > 30):
            return True
    if epoch == 160:
        if (best_IS < 8) or (best_fid > 20):
            return True

    return False



