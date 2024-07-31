# CAGAN

The search code will be published once the paper is accepted, and the training code and network weights will be published immediately.
## Code used for "CAGAN: Constrained Neural Architecture Search for GANs". This paper was accepted in the KBS Journal 2024. 

# Abstract
Recently, a number of Neural Architecture Search (NAS) methods have been proposed to automate the design of Generative Adversarial Networks (GANs). However, due to the unstable training of GANs and the multi-model forgetting of one-shot NAS, the stability of embedding NAS into GANs is still not satisfactory. Thus, we propose a constrained evolutionary NAS method for GANs (called CAGAN) and design a first benchmark (NAS-GAN-Bench-101) for NAS in GANs. First, we constrain the sampling architecture size to steer the evolutionary search towards more promising and lightweight architectures. Subsequently, we propose a shape-constrained sampling strategy to select more reasonable architectures. Moreover, we present a multi-objective decomposition selection strategy to simultaneously consider the model shape, Inception Score (IS), and Fréchet Inception Distance (FID), which produces diverse superior generator candidates. CAGAN has been applied to unconditioned image generation tasks, in which the evolutionary search of GANs on the CIFAR-10 is completed in 0.35 GPU days. Our searched GANs showed promising results on the CIFAR-10 with (IS=8.96±0.06, FID=9.45) and surpassed previous NAS-designed GANs on the STL-10 with (IS=10.39±0.13, FID=19.34)

# How ot set up CAGAN.  
The environment of CAGAN is more complex, training and searching are torch-based, but part of the evaluation needs to call the api of TensorFlow 2. For better reading, we provide English tutorials and 中文 tutorials.
## 1. Environment requirements:
The search environment is consistent with AlphaGAN，to run this code, you need:  
- PyTorch 2.0
- TensorFlow 2.12
- cuda 12.0  

Other requirements are in environment.yaml 

<!-- install code  -->
<pre><code>conda env create -f environment.yaml
</code></pre>

## 2.prepare fid statistic file
you need to create "fid_stat" directory and download the statistical files of real images.
<pre><code>mkdir fid_stat
</code></pre>


# unfinished and to be continued\ 代码和教程还在更新中

# 中文版运行教程 
同时配置torch和TensorFlow有一定难度。
## 1. 环境配置:
### 1.1 基础要求
CAGAN的搜索环境用的是最新的pytorch2.0以上的版本，配合TensorFlow2.12以上的版本。
- python=3.11
- pytorch=2.0.1
- tensorflow=2.12.0
- tensorflow-gan=2.1.0

### 1.2 百度网盘环境
考虑到同时安装配置torch和TensorFlow有一定难度，我们在百度网盘准备了已经配置好的安装包。
链接: https://pan.baidu.com/s/1I_3zXugfGJAg6l5PEdsV_w 提取码: 83gs 
下载完毕后，直接解压到'/home/user/anaconda3/envs/'之下即可，文件目录如下所示。
<pre><code>
cd /home/yangyeming/anaconda3/envs
tar -xvf torch.tar
</code></pre>
![image](https://github.com/user-attachments/assets/c85ea01b-ac3b-4b81-8fea-a8e990af247b)

随后，激活相关环境。
<pre><code>
cd CAGAN
conda activate torch 
</code></pre>

### 1.3 数据集准备(Cifar-10 and STL-10)
在CAGAN中，我们用CIFAR-10和STL-10数据集用来评价。默认数据集存放在 ./datasets/cifar10 and ./datasets/stl10.
读者可以自行下载，或者由data代码自动下载。 

## 1.4. 准备fid_stat和tmp的文件夹
需要从EAGAN（https://github.com/marsggbo/EAGAN）中下载相关数据。
<pre><code>mkdir fid_stat
mkdir tmp
</code></pre>


# Acknowledgement
Some of the codes are built by:

1. [MMD-AdversarialNAS](https://ieeexplore.ieee.org/document/10446488)

2. [EAGAN](https://github.com/marsggbo/EAGAN)

3. [AlphaGAN](https://github.com/yuesongtian/AlphaGAN)

Thanks them for their great works!
