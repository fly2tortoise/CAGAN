# @Date    : 2020-12-30
# @Author  : Guohao Ying

"""GanAlgorithm."""
from __future__ import absolute_import, division, print_function
import copy
import os
import numpy as np
import random
from utils.utils import count_parameters_in_MB, count_parameters_search_Arch_shape_MB
from archs.fully_super_network import Discriminator, Generator
from utils.FlopWithReturn import return_FLOPs
#To implement easy, wo use int number in the code to represent the 
# corresponding operations. These are replaced with one-hot in the paper,
# which can better explain our theory. Each int number is equal to an one-hot number.

# 编码方式

class GanAlgorithm():

    def __init__(self, args):
        self.args = args
        self.steps = 3
        self.up_nodes = 2
        self.down_nodes = 2
        self.normal_nodes = 5
        self.dis_normal_nodes = 5        # steps= 3  normal =5  up down =2
        self.archs = {}
        self.dis_archs = {}
        self.base_dis_arch = np.array(
            [[3, 0, 3, 0, 2, 0, 0], [3, 0, 3, 0, 1, -1, -1], [3, 0, 3, 0, 1, -1, -1]])
        # init model, but this is the best code for discriminator
        # I convince the sampling code is [[3, None, 3, None, 2, 0, 0], [3, None, 3, None, 1, -1, -1], [3, None, 3, None, 1, -1, -1]]
        # Best D is: [[3, 0, 3, 0, 2, 0, 0], [1, 0, 3, 0, 1, -1, -1], [3, 0, 3, 0, 1, -1, -1]]

    def encode(self, genotype):
        lists = [0 for i in range(self.steps)]  # 3
        for i in range(len(lists)):
            lists[i] = str(genotype[i])
        return tuple(lists)                     # list 3cell  == a sub-net

    def search(self):
        # 返回生成器的编码
        new_genotype = self.sample()            # new_genotype(list) == a sub-net
        # print(new_genotype)
        t = self.encode(new_genotype)           # list(int) -> tuple(str)
        while(t in self.archs):
            new_genotype = self.sample()
            t = self.encode(new_genotype)       # t tuple , archs[t] list  new_gen list
        self.archs[t] = new_genotype
        return new_genotype

    def search_constraint(self):
        # 返回生成器的编码
        new_genotype = self.sample_constraint()            # new_genotype(list) == a sub-net
        # print(new_genotype)
        t = self.encode(new_genotype)           # list(int) -> tuple(str)
        while(t in self.archs):
            new_genotype = self.sample_constraint()
            t = self.encode(new_genotype)       # t tuple , archs[t] list  new_gen list
        self.archs[t] = new_genotype
        return new_genotype

    def search_dis(self):
        # 返回鉴别器的编码
        new_genotype = self.sample_D()
        t = self.encode(new_genotype)
        while(t in self.dis_archs):
            new_genotype = self.sample_D()
            t = self.encode(new_genotype)
        self.dis_archs[t] = new_genotype
        return new_genotype

    def sample(self):
        genotype = np.zeros(
            (self.steps, self.up_nodes+self.normal_nodes), dtype=int)  # (3, 2+5)
        for i in range(self.steps):         # 3 cell
            for j in range(self.up_nodes):  # 2 up
                genotype[i][j] = random.randint(0, 2)
            while(genotype[i][2] == 0 and genotype[i][3] == 0):  # 未赋值
                for k in range(2):
                    genotype[i][k+2] = random.randint(0, 6)
            while(genotype[i][4] == 0 and genotype[i][5] == 0 and genotype[i][6] == 0):  # 4 5 6 号均为 none 未连接需要重赋
                for k in range(2, self.normal_nodes):           # 2 - 5
                    genotype[i][k+2] = random.randint(0, 6)     # 有一个不为0 网络即可继续
        return genotype

    def sample_constraint(self):
        genotype = np.zeros((self.steps, self.up_nodes + self.normal_nodes), dtype=int)  # (3, 2 +5)
        cnt = 0
        while (cnt == 0):
            genotype = np.zeros((self.steps, self.up_nodes + self.normal_nodes), dtype=int)  # (3, 2 +5)
            for i in range(self.steps):  # 3 cell
                for j in range(self.up_nodes):  # 2 up
                    genotype[i][j] = random.randint(0, 2)
                while (genotype[i][2] == 0 and genotype[i][3] == 0):  # 未赋值
                    for k in range(2):
                        genotype[i][k + 2] = random.randint(0, 6)
                while (genotype[i][4] == 0 and genotype[i][5] == 0 and genotype[i][6] == 0):  # 4 5 6 号均为 none 未连接需要重赋
                    for k in range(2, self.normal_nodes):  # 2 - 5
                        genotype[i][k + 2] = random.randint(0, 6)  # 有一个不为0 网络即可继续

            Generator_subnet = Generator(self.args, genotype)
            param_szie = count_parameters_in_MB(Generator_subnet)
            param_cell_szie, _ = count_parameters_search_Arch_shape_MB(Generator_subnet)
            FlopsG = return_FLOPs(Generator_subnet, (1, self.args.latent_dim))
            # print(genotype.tolist(), param_szie, FlopsG)
            if (FlopsG >= 1500 and FlopsG <= 3000) and (param_szie >= 6.5 and param_szie <= 11.5) and (param_cell_szie[0] <= 2.0) and (param_cell_szie[1] <= 2.5) and (param_cell_szie[2] <= 2.5):
                print(param_szie, FlopsG, genotype.tolist())  # test the Flops\ for G
                cnt = 1

        return genotype
        
    def sample_D(self):
        genotype = np.zeros((self.steps, self.down_nodes +
                             self.normal_nodes), dtype=int)    # (3, 2+5)
        for i in range(self.steps):
            genotype[i][0] = random.randint(1, 6)                       # 0 号 头部
            while(genotype[i][1] == 0 and genotype[i][2] == 0):
                for k in range(2):
                    genotype[i][k+2] = random.randint(0, 6)
            while(genotype[i][3] == 0 and genotype[i][4] == 0):         # 中部
                for k in range(2, 4):
                    genotype[i][k+1] = random.randint(0, 6)
            genotype[i][self.dis_normal_nodes] = random.randint(-1, 5)  # 尾部 下采样
            if genotype[i][self.dis_normal_nodes] == -1:
                genotype[i][self.dis_normal_nodes+1] = -1               # -1 有什么用吗 ？
            else:
                genotype[i][self.dis_normal_nodes+1] = random.randint(0, 5)   # 下采样 有6种
        return genotype

    def sample_Flops_D(self):
        genotype = np.zeros((self.steps, self.down_nodes +
                             self.normal_nodes), dtype=int)  # (3, 2+5)
        cnt = 0
        while(cnt == 0):
            genotype = np.zeros((self.steps, self.down_nodes + self.normal_nodes), dtype=int)  # (3, 2+5)
            for i in range(self.steps):
                genotype[i][0] = random.randint(1, 6)                       # 0 号 头部
                while(genotype[i][1] == 0 and genotype[i][2] == 0):
                    for k in range(2):
                        genotype[i][k+2] = random.randint(0, 6)
                while(genotype[i][3] == 0 and genotype[i][4] == 0):         # 中部
                    for k in range(2, 4):
                        genotype[i][k+1] = random.randint(0, 6)
                genotype[i][self.dis_normal_nodes] = random.randint(-1, 5)  # 尾部 下采样
                if genotype[i][self.dis_normal_nodes] == -1:
                    genotype[i][self.dis_normal_nodes+1] = -1               # -1 有什么用吗 ？
                else:
                    genotype[i][self.dis_normal_nodes+1] = random.randint(0, 5)   # 下采样 有6种

            # print(genotype.tolist())
            dis_Flops = Discriminator(self.args, genotype)
            FlopsD = return_FLOPs(dis_Flops, (1, 3, self.args.img_size, self.args.img_size))

            if FlopsD >= 150 and FlopsD <= 300:
                print("Init_D: ", FlopsD, genotype.tolist())  # test the Flops for D, the standers is 263MB
                cnt = 1

        return genotype


    def judge_repeat(self, new_genotype):
        t = self.encode(new_genotype)
        return t in self.archs

    def judge_repeat_dis(self, new_genotype):
        t = self.encode(new_genotype)
        return t in self.dis_archs

    # def search_mutate(self):
    #     t = self.encode(self.base_gen_arch)
    #     self.archs[t] = self.base_gen_arch
    #     while(t in self.archs):
    #         new_genotype = self.mutation_gen(self.base_gen_arch)
    #         t = self.encode(new_genotype)
    #     self.archs[t] = new_genotype
    #     return new_genotype

    def search_mutate_dis(self):
        t = self.encode(self.base_dis_arch)        # tuple
        self.dis_archs[t] = self.base_dis_arch     # list
        while(t in self.dis_archs):
            new_genotype = self.mutation_gen(self.base_dis_arch)  # mutation gen
            t = self.encode(new_genotype)
        self.dis_archs[t] = new_genotype
        return new_genotype

    def mutation_dis(self, alphas_a, ratio=0.5):
        """Mutation for an individual"""
        new_alphas = alphas_a.copy()
        layer = random.randint(0, self.steps-1)
        index = random.randint(0, self.down_nodes+self.dis_normal_nodes-1)
        if index == 0:
            new_alphas[layer][index] = random.randint(1, 6)
            while(new_alphas[layer][index] == alphas_a[layer][index]):
                new_alphas[layer][index] = random.randint(1, 6)
        elif index >= 1 and index < 3:
            new_alphas[layer][index] = random.randint(0, 6)
            while (new_alphas[layer][1] == 0 and new_alphas[layer][2] == 0) or (new_alphas[layer][index] == alphas_a[layer][index]):
                new_alphas[layer][index] = random.randint(0, 6)
        elif index >= 3 and index < 5:
            new_alphas[layer][index] = random.randint(0, 6)
            while(new_alphas[layer][3] == 0 and new_alphas[layer][4] == 0) or (new_alphas[layer][index] == alphas_a[layer][index]):
                new_alphas[layer][index] = random.randint(0, 6)
        if index == 5:
            new_alphas[layer][index] = random.randint(-1, 5)
            while(new_alphas[layer][index] == alphas_a[layer][index]):
                new_alphas[layer][index] = random.randint(-1, 5)
        if index == 6:
            new_alphas[layer][index] = random.randint(0, 5)
            while(new_alphas[layer][index] == alphas_a[layer][index]):
                new_alphas[layer][index] = random.randint(0, 5)
        return new_alphas

    def mutation_gen(self, alphas_a, ratio=0.5):
        """Mutation for an individual"""
        new_alphas = alphas_a.copy()
        layer = random.randint(0, self.steps-1)
        index = random.randint(0, self.down_nodes+self.normal_nodes-1)
        if index < 2:
            new_alphas[layer][index] = random.randint(0, 2)
            while(new_alphas[layer][index] == alphas_a[layer][index]):
                new_alphas[layer][index] = random.randint(0, 2)
        elif index >= 2 and index < 4:
            new_alphas[layer][index] = random.randint(0, 6)
            while (new_alphas[layer][2] == 0 and new_alphas[layer][3] == 0) or (new_alphas[layer][index] == alphas_a[layer][index]):
                new_alphas[layer][index] = random.randint(0, 6)
        elif index >= 4:
            new_alphas[layer][index] = random.randint(0, 6)
            while(new_alphas[layer][3] == 0 and new_alphas[layer][4] == 0) or (new_alphas[layer][index] == alphas_a[layer][index]):
                new_alphas[layer][index] = random.randint(0, 6)
        return new_alphas
