# CAGAN

The search code will be published once the paper is accepted, and the training code and network weights will be published immediately.
## Code used for "CAGAN: Constrained Neural Architecture Search for GANs". This paper was published in the KBS Journal 2024. 

# Introduction
Recently, a number of Neural Architecture Search (NAS) methods have been proposed to automate the design of Generative Adversarial Networks (GANs). However, due to the unstable training of GANs and the multi-model forgetting of one-shot NAS, the stability of embedding NAS into GANs is still not satisfactory. Thus, we propose a constrained evolutionary NAS method for GANs (called CAGAN) and design a first benchmark (NAS-GAN-Bench-101) for NAS in GANs. First, we constrain the sampling architecture size to steer the evolutionary search towards more promising and lightweight architectures. Subsequently, we propose a shape-constrained sampling strategy to select more reasonable architectures. Moreover, we present a multi-objective decomposition selection strategy to simultaneously consider the model shape, Inception Score (IS), and Fréchet Inception Distance (FID), which produces diverse superior generator candidates. CAGAN has been applied to unconditioned image generation tasks, in which the evolutionary search of GANs on the CIFAR-10 is completed in 0.35 GPU days. Our searched GANs showed promising results on the CIFAR-10 with (IS=8.96±0.06, FID=9.45) and surpassed previous NAS-designed GANs on the STL-10 with (IS=10.39±0.13, FID=19.34)

# Set Up 
## 0. The environment of CAGAN is more complex, training and searching are torch-based, but part of the evaluation needs to call the api of TensorFlow 2. For better reading, we provide English tutorials and 中文 tutorials.
## 1. Environment requirements:
The search environment is consistent with AlphaGAN，to run this code, you need:  
- PyTorch 2.1  
- TensorFlow 2.15 
- cuda 12.0  

Other requirements are in environment.yaml 

<!-- install code  -->
<pre><code>conda env create -f environment.yaml
</code></pre>

## 2.prepare fid statistic file
you need to create "fid_stat" directory and download the statistical files of real images.
<pre><code>mkdir fid_stat
</code></pre>


# unfinished and to be continued

# 中文版运行教程 
## 1. 环境配置:
### 1.1 基础要求
CAGAN的搜索环境用的是最新的pytorch2.0以上的版本，配合TensorFlow2.13之后的版本。
- PyTorch 2.1  
- TensorFlow 2.15 
- cuda 12.0  

### 1.2 百度网盘环境
考虑到同时安装配置torch和TensorFlow有一定难度，我们在百度网盘准备了已经配置好的安装包，直接解压即可。

### 1.3 代码补全
按照随后流程补齐代码。


## 2. 准备fid的统计数据文件
you need to create "fid_stat" directory and download the statistical files of real images.
<pre><code>mkdir fid_stat
</code></pre>


# Acknowledgement
Some of the codes are built by:

1.[MMD-AdversarialNAS](https://ieeexplore.ieee.org/document/10446488)

2.[EAGAN](https://github.com/marsggbo/EAGAN)

3.[AlphaGAN](https://github.com/yuesongtian/AlphaGAN)

Thanks them for their great works!
