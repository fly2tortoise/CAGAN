
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from imageio import imsave
from tqdm import tqdm
from copy import deepcopy
import logging
import random
import heapq
from scipy.stats import spearmanr
from utils.sort import CARS_NSGA
from utils.utils import count_parameters_in_MB, count_parameters_search_Arch_shape_MB, count_cosine_similarity_3cell, count_cosine_similarity_6layer,calculate_similarity_score
# from utils.inception_score import get_inception_score
# from utils.fid_score import calculate_fid_given_paths
from Tf2Eval import InceptionScore
from pytorch_fid.fid_score import calculate_fid_given_paths
from archs.fully_super_network import Generator
from utils.MOEAD_sort import Prefer_MOEAD
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class GenTrainer():
    def __init__(self, args, gen_net, dis_net, gen_optimizer, dis_optimizer, train_loader, gan_alg, dis_genotype):
        self.args = args
        self.gen_net = gen_net
        self.dis_net = dis_net
        self.gen_optimizer = gen_optimizer
        self.dis_optimizer = dis_optimizer
        self.train_loader = train_loader
        self.gan_alg = gan_alg
        self.dis_genotype = dis_genotype
        # self.genotypes = np.stack([gan_alg.search() for i in range(args.num_individual)], axis=0)
        self.genotypes = np.stack([gan_alg.search_constraint() for i in range(args.num_individual)],
                                  axis=0)  # 随机生成 genotypes 堆叠起来

    def train(self, epoch, writer_dict, schedulers=None):
        writer = writer_dict['writer']
        gen_step = 0

        # train mode 传统的训练模型技术
        gen_net = self.gen_net.train()
        dis_net = self.dis_net.train()

        for iter_idx, (imgs, _) in enumerate(tqdm(self.train_loader)):
            global_steps = writer_dict['train_global_steps']

            real_imgs = imgs.type(torch.cuda.FloatTensor)
            i = np.random.randint(0, self.args.num_individual, 1)[0]
            if epoch <= self.args.warmup:
                genotype_G = self.gan_alg.search()
            else:
                genotype_G = self.genotypes[i]
            # sample noise
            z = torch.cuda.FloatTensor(np.random.normal(
                0, 1, (imgs.shape[0], self.args.latent_dim)))
            # train D
            self.dis_optimizer.zero_grad()
            # real_validity = dis_net(real_imgs, genotype_D)
            real_validity = dis_net(real_imgs)
            fake_imgs = gen_net(z, genotype_G).detach()
            assert fake_imgs.size() == real_imgs.size()
            # fake_validity = dis_net(fake_imgs, genotype_D)
            fake_validity = dis_net(fake_imgs)
            # Hinge loss
            d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + \
                torch.mean(nn.ReLU(inplace=True)(1 + fake_validity))
            d_loss.backward()
            self.dis_optimizer.step()
            writer.add_scalar('d_loss', d_loss.item(), global_steps)
            # train G
            if global_steps % self.args.n_critic_search == 0:
                self.gen_optimizer.zero_grad()
                # sample noise
                gen_z = torch.cuda.FloatTensor(np.random.normal(
                    0, 1, (self.args.gen_bs, self.args.latent_dim)))

                gen_imgs = gen_net(gen_z, genotype_G)
                # fake_validity = dis_net(gen_imgs, genotype_D)
                fake_validity = dis_net(gen_imgs)
                # Hinge loss
                g_loss = -torch.mean(fake_validity)
                g_loss.backward()
                self.gen_optimizer.step()
                # learning rate
                if schedulers:
                    gen_scheduler, dis_scheduler = schedulers
                    g_lr = gen_scheduler.step(global_steps)
                    d_lr = dis_scheduler.step(global_steps)
                    writer.add_scalar('LR/g_lr', g_lr, global_steps)
                    writer.add_scalar('LR/d_lr', d_lr, global_steps)
                writer.add_scalar('g_loss', g_loss.item(), global_steps)
                gen_step += 1
            # verbose
            if gen_step and iter_idx % self.args.print_freq == 0:
                tqdm.write(
                    '[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]' %
                    (epoch, self.args.max_epoch_D, iter_idx % len(self.train_loader), len(self.train_loader), d_loss.item(), g_loss.item()))
            writer_dict['train_global_steps'] = global_steps + 1

    def search_evol_arch(self, epoch, fid_stat):
        # search_evol_arch包含种群 约束初始化、约束交叉变异、架构性能评价、差分选择机制，最终返回的是优质的架构
        # 1. 初始化  2. 交叉变异
        offsprings = self.gen_offspring(self.genotypes)
        genotypes = np.concatenate((self.genotypes, offsprings), axis=0)
        # 创建评价list
        is_values, fid_values, params, params_fitness = np.zeros(len(genotypes)), np.zeros(
            len(genotypes)), np.zeros(len(genotypes)), np.zeros(len(genotypes))
        keep_N, selected_N = len(offsprings), self.args.num_selected

        # 3. 架构性能评价，其中IS和FID是通过生成图片评价，而架构形状相似度S是通过先验知识设定来评价的。
        for idx, genotype_G in enumerate(tqdm(genotypes)):
            # 计算架构生成图片清晰度和分布差距
            is_value, is_std, fid_value = self.validate(genotype_G, fid_stat)
            param_szie = count_parameters_in_MB(Generator(self.args, genotype_G))
            # 计算架构参数形状相似度，分别计算layer级形状和cell级别的形状
            Param_3_layer, Param_6_layer = count_parameters_search_Arch_shape_MB(Generator(self.args, genotype_G))
            param_fit_3 = count_cosine_similarity_3cell(Param_3_layer)
            param_fit_6 = count_cosine_similarity_6layer(Param_6_layer)
            param_fit_all = calculate_similarity_score(reference_value_min=8.5, unknown_value = param_szie, reference_value_max= 10)
            # print(param_fit_all)
            param_fit = (param_fit_all + param_fit_3 + param_fit_6 ) / 3
            # print("generator:", genotype_G.tolist())

            is_values[idx] = is_value
            fid_values[idx] = fid_value
            params[idx] = param_szie
            params_fitness[idx] = param_fit

            logger.info(f'generator: {genotype_G.tolist()}.')
            logger.info(f'is_value: {is_value}, fid_value: {fid_value}, @ idx {idx}.')
            logger.info(f'param_szie: {param_szie}, param_fit: {param_fit},@ idx {idx}.')
            # logger.info(f'param_szie: {param_szie}, param_fit: {param_fit},@ idx {idx}.')
            # print(idx,"Arch performance is：",is_value, fid_value, param_szie, param_fit)

        # 绘制当前种群图片
        num_colors = 20
        start_color = np.array([0.8, 0.8, 1.0])  # 起始颜色（RGB值，这里为浅蓝色）
        mid_color = np.array([0.1, 0.1, 0.8])  # 中间颜色（RGB值，这里为深蓝色）
        end_color = np.array([0.0, 0.0, 0.0])  # 结束颜色（RGB值，这里为黑色）
        # 创建颜色渐变序列
        colors = np.concatenate([
            np.linspace(start_color, mid_color, num_colors // 2),
            np.linspace(mid_color, end_color, num_colors - num_colors // 2)
        ])
        x = is_values
        y = fid_values
        # 按照标签取颜色
        lable = 0
        lable_num = int((self.args.warmup) / 10)
        if (epoch / 10 >= 0):
            lable = int(epoch / 10) - lable_num
        # plt.figure(figsize=(8, 8))  # plot in same PDF
        plt.title('Each 10 epoch population performance')
        for i in range(64):
            plt.scatter(y[i], 1 / x[i], color=colors[lable])
        plt.xlabel('FID values')
        plt.ylabel('1/IS values')
        dir1 = self.args.path_helper['graph_vis_path']
        plt.savefig(dir1 + str(epoch) + "values.pdf", dpi=750, bbox_inches='tight', format='pdf')

        # 打印当前性能变化
        logger.info(f'mean_IS_values: {np.mean(is_values)}, mean_FID_values: {np.mean(fid_values)},@ epoch {epoch}.')
        logger.info(f'MXA_IS_values: {np.max(is_values)}, MIN_FID_values: {np.min(fid_values)},@ epoch {epoch}.')
        logger.info(f'mean_IS_front32: {np.mean(is_values[:32])}, mean_FID_front32: {np.mean(fid_values[:32])},@ epoch {epoch}.')
        logger.info(f'MXA_IS_front32: {np.max(is_values[:32])}, MIN_FID_front32: {np.min(fid_values[:32])},@ epoch {epoch}.')

        # 如果发现所有架构的平均性能都很低，即超网训练失败，则立即终止程序，这种情况较少出现。
        if np.mean(fid_values) >450 and np.max(is_values) <1.3:
            import sys
            sys.exit("mode collapse encountered. Exiting program.")

        # EAGAN采用的是CARS保护性非支配排序选择
        # obj = [fid_values, 1 / params_fitness]
        # keep, selected = CARS_NSGA(is_values, obj, keep_N), CARS_NSGA(is_values, obj, selected_N)

        # CAGAN采用的是MOEAD差分选择
        print("uniform MOEAD to sort")
        is_values_opposite = 1 / is_values
        params_fitness_opposite = 1 / params_fitness
        random_F = np.zeros((64, 3))
        random_F[:, 0] = is_values_opposite
        random_F[:, 1] = fid_values
        random_F[:, 2] = params_fitness_opposite
        weights = np.array([3 / 10, 3 / 10, 3 / 10])
        keep, selected = Prefer_MOEAD(random_F, keep_N, weights), Prefer_MOEAD(random_F, selected_N, weights)

        # 记录选择后 top-8个架构的性能
        selected_IS = []
        selected_FID = []
        selected_params = []
        for i in selected:   # 8
            logger.info(f'genotypes_{i}, ISs: {is_values[i]}, FIDs: {fid_values[i]}, Param: {params[i]}|| @ epoch {epoch}.')
            selected_IS.append(is_values[i])
            selected_FID.append(fid_values[i])
            selected_params.append(params[i])
        # 记录选择的架构性能平均值
        logger.info(f'mean_IS_select8: {np.mean(selected_IS)}, mean_FID_select8: {np.mean(selected_FID)},mean_params_select8: {np.mean(selected_params)}.')
        logger.info(f'MXA_IS_select8: {np.max(selected_IS)}, MIN_FID_select8: {np.min(selected_FID)},MIN_params_select8: {np.min(selected_params)}.')
        self.genotypes = genotypes[keep]
        logger.info(genotypes[selected].tolist())

        return genotypes[selected]

    def validate(self, genotype_G, fid_stat):
        # eval mode
        gen_net = self.gen_net.eval()
        # get fid and inception score
        fid_buffer_dir = os.path.join(
            self.args.path_helper['sample_path'], 'fid_buffer')
        os.makedirs(fid_buffer_dir, exist_ok=True)
        eval_iter = self.args.num_eval_imgs // self.args.eval_batch_size
        img_list = list()
        for iter_idx in tqdm(range(eval_iter), desc='sample images'):
            z = torch.cuda.FloatTensor(np.random.normal(
                0, 1, (self.args.eval_batch_size, self.args.latent_dim)))
            # generate a batch of images
            gen_imgs = gen_net(z, genotype_G).mul_(127.5).add_(127.5).clamp_(
                0.0, 255.0).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
            for img_idx, img in enumerate(gen_imgs):
                file_name = os.path.join(
                    fid_buffer_dir, f'iter{iter_idx}_b{img_idx}.png')
                imsave(file_name, img)
            img_list.extend(list(gen_imgs))

        # get inception score
        logger.info('=> calculate inception score')
        mean, std = InceptionScore.get_inception_score(img_list)
        # get fid score
        logger.info('=> calculate fid score')
        device = torch.device('cuda')
        fid_score = calculate_fid_given_paths(paths=[fid_buffer_dir, fid_stat], batch_size=50, device=device, dims=2048,
                                              num_workers=16)

        return mean, std, fid_score

    def gen_offspring(self, alphas, offspring_ratio=1.0):
        """Generate offsprings.
        :param alphas: Parameteres for populations
        :type alphas: nn.Tensor
        :param offspring_ratio: Expanding ratio
        :type offspring_ratio: float
        :return: The generated offsprings
        :rtype: nn.Tensor
        """
        n_offspring = int(offspring_ratio * alphas.shape[0])
        offsprings = []
        while len(offsprings) != n_offspring:
            rand = np.random.rand()
            if rand < 0.5:
                alphas_c = self.mutation(
                    alphas[np.random.randint(0, alphas.shape[0])])
            else:
                a, b = np.random.randint(
                    0, alphas.shape[0]), np.random.randint(0, alphas.shape[0])
                while(a == b):
                    a, b = np.random.randint(
                        0, alphas.shape[0]), np.random.randint(0, alphas.shape[0])
                alphas_c = self.crossover(alphas[a], alphas[b])
            if not self.gan_alg.judge_repeat(alphas_c):
                offsprings.append(alphas_c)
        offsprings = np.stack(offsprings, axis=0)
        return offsprings

    def judge_repeat(self, alphas, new_alphas):
        """Judge if two individuals are the same.
        :param alphas_a: An individual
        :type alphas_a: nn.Tensor
        :param new_alphas: An individual
        :type new_alphas: nn.Tensor
        :return: True or false
        :rtype: nn.Tensor
        """
        diff = np.reshape(np.absolute(
            alphas - np.expand_dims(new_alphas, axis=0)), (alphas.shape[0], -1))
        diff = np.sum(diff, axis=1)
        return np.sum(diff == 0)

    def crossover(self, alphas_a, alphas_b):
        """Crossover for two individuals."""
        # alpha a
        new_alphas = alphas_a.copy()
        # alpha b
        layer = random.randint(0, 2)
        index = random.randint(0, 6)
        while(new_alphas[layer][index] == alphas_a[layer][index]):
            layer = random.randint(0, 2)
            index = random.randint(0, 6)
            new_alphas[layer][index] = alphas_b[layer][index]
            if index >= 2 and index < 4 and new_alphas[layer][2] == 0 and new_alphas[layer][3] == 0:
                new_alphas[layer][index] = alphas_a[layer][index]
            elif index >= 4 and new_alphas[layer][4] == 0 and new_alphas[layer][5] == 0 and new_alphas[layer][6] == 0:
                new_alphas[layer][index] = alphas_a[layer][index]
        return new_alphas

    def mutation(self, alphas_a, ratio=0.5):
        """Mutation for An individual."""
        new_alphas = alphas_a.copy()
        layer = random.randint(0, 2)
        index = random.randint(0, 6)
        if index < 2:
            new_alphas[layer][index] = random.randint(0, 2)
            while(new_alphas[layer][index] == alphas_a[layer][index]):
                new_alphas[layer][index] = random.randint(0, 2)
        elif index >= 2 and index < 4:
            new_alphas[layer][index] = random.randint(0, 6)
            while(new_alphas[layer][2] == 0 and new_alphas[layer][3] == 0) or (new_alphas[layer][index] == alphas_a[layer][index]):
                new_alphas[layer][index] = random.randint(0, 6)
        elif index >= 4:
            new_alphas[layer][index] = random.randint(0, 6)
            while(new_alphas[layer][4] == 0 and new_alphas[layer][5] == 0 and new_alphas[layer][6] == 0) or (new_alphas[layer][index] == alphas_a[layer][index]):
                new_alphas[layer][index] = random.randint(0, 6)
        return new_alphas

    def select_best(self, epoch):
        values = []
        for genotype_G in self.genotypes:
            ssim_value, psnr_value = self.validate(genotype_G)
            #logger.info(f'ssim_value: {ssim_value}, psnr_value: {psnr_value}|| @ epoch {epoch}.')
            values.append(ssim_value)
        max_index = values.index(max(values))
        return self.genotypes[max_index]


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def copy_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten


class LinearLrDecay(object):
    def __init__(self, optimizer, start_lr, end_lr, decay_start_step, decay_end_step):

        assert start_lr > end_lr
        self.optimizer = optimizer
        self.delta = (start_lr - end_lr) / (decay_end_step - decay_start_step)
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_end_step
        self.start_lr = start_lr
        self.end_lr = end_lr

    def step(self, current_step):
        if current_step <= self.decay_start_step:
            lr = self.start_lr
        elif current_step >= self.decay_end_step:
            lr = self.end_lr
        else:
            lr = self.start_lr - self.delta * \
                (current_step - self.decay_start_step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        return lr
