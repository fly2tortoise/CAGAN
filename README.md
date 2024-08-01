# CAGAN

The search code will be published once the paper is accepted, and the training code and network weights will be published immediately.
## Code used for "CAGAN: Constrained Neural Architecture Search for GANs". This paper was accepted in the KBS Journal 2024. 

# Abstract
Recently, a number of Neural Architecture Search (NAS) methods have been proposed to automate the design of Generative Adversarial Networks (GANs). However, due to the unstable training of GANs and the multi-model forgetting of one-shot NAS, the stability of embedding NAS into GANs is still not satisfactory. Thus, we propose a constrained evolutionary NAS method for GANs (called CAGAN) and design a first benchmark (NAS-GAN-Bench-101) for NAS in GANs. First, we constrain the sampling architecture size to steer the evolutionary search towards more promising and lightweight architectures. Subsequently, we propose a shape-constrained sampling strategy to select more reasonable architectures. Moreover, we present a multi-objective decomposition selection strategy to simultaneously consider the model shape, Inception Score (IS), and Fréchet Inception Distance (FID), which produces diverse superior generator candidates. CAGAN has been applied to unconditioned image generation tasks, in which the evolutionary search of GANs on the CIFAR-10 is completed in 0.35 GPU days. Our searched GANs showed promising results on the CIFAR-10 with (IS=8.96±0.06, FID=9.45) and surpassed previous NAS-designed GANs on the STL-10 with (IS=10.39±0.13, FID=19.34)

# How ot set up CAGAN.  
The environment of CAGAN is more complex, training and searching are torch-based, but part of the evaluation needs to call the api of TensorFlow 2. For better reading, we provide English tutorials and 中文 tutorials.

## 1. Environment requirements:
### 1.1 Basic Requirements
CAGAN's search environment uses the latest version of PyTorch 2.0 and above, along with TensorFlow 2.12 and above.
- python=3.11
- pytorch=2.0.1
- tensorflow=2.12.0
- tensorflow-gan=2.1.0

### 1.2 Baidu Cloud Environment
Considering the difficulty of simultaneously installing and configuring Torch and TensorFlow, we have prepared pre-configured installation packages on Baidu Cloud.

Link: https://pan.baidu.com/s/1I_3zXugfGJAg6l5PEdsV_w
Access Code: 83gs

After downloading, simply extract it to '/home/user/anaconda3/envs/'. The file directory is as follows.
<pre><code>cd /home/yangyeming/anaconda3/envs
tar -xvf torch.tar
</code></pre>
![image](https://github.com/user-attachments/assets/c85ea01b-ac3b-4b81-8fea-a8e990af247b)

Then, activate the relevant environment.
<pre><code>cd CAGAN
conda activate torch 
</code></pre>

### 1.3 Dataset Preparation (CIFAR-10 and STL-10)
In CAGAN, we use the CIFAR-10 and STL-10 datasets for evaluation. The default datasets are stored in ./datasets/cifar10 and ./datasets/stl10.
Readers can download them manually or use the data code to download them automatically.

### 1.4 Preparing the fid_stat and tmp Folders
You need to download the relevant data from EAGAN. https://github.com/marsggbo/EAGAN
<pre><code>mkdir fid_stat
mkdir tmp
</code></pre>

## 2. Architecture Search
### 2.1 Constraint Architecture Search to Design GANs
<pre><code>bash train_search_gen.sh
</code></pre> 
### 2.2 Fully Train the Searched GANs Using the Hinge-loss Function
<pre><code>bash train_arch_cifar10.sh
bash train_arch_stl10.sh
</code></pre>
### 2.3 Fully Train the Searched GANs Using the MMD-GAN Loss Function
We used the training environment provided by MMD-AdversarialNAS and found that the networks trained with the MMD-loss performed well. In this step, you only need to replace the training architecture of MMD-AdversarialNAS with the one found by CAGAN.
<pre><code>bash train_arch_cifar10.sh
bash train_arch_stl10.sh
</code></pre>

# unfinished and to be continued



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

### 1.4. 准备fid_stat和tmp的文件夹
需要从EAGAN中下载相关数据。https://github.com/marsggbo/EAGAN
<pre><code>mkdir fid_stat
mkdir tmp
</code></pre>
<pre><code>mkdir fid_stat
mkdir tmp
</code></pre>

## 2. 架构搜索
### 2.1 约束架构搜索设计GANs
<pre><code>bash train_search_gen.sh
</code></pre> 

### 2.2 使用Hing-loss的损失函数充分训练搜索后的GANs
<pre><code>bash train_arch_cifar10.sh
bash train_arch_stl10.sh
</code></pre>

### 2.3 使用MMD-GAN的损失函数充分训练搜索后的GANs
我们使用了MMD-AdversarialNAS给出的训练环境，发现经过MMD-loss训练后的网络，效果良好。这一步只需要将MMD-AdversarialNAS的训练部分的架构更换为CAGAN搜索到的即可。
<pre><code> bash train_arch_cifar10.sh
bash train_arch_stl10.sh
</code></pre>

# 代码和教程还在更新中

# Acknowledgement
Some of the codes are built by:

1. [MMD-AdversarialNAS](https://ieeexplore.ieee.org/document/10446488)

2. [EAGAN](https://github.com/marsggbo/EAGAN)

3. [AlphaGAN](https://github.com/yuesongtian/AlphaGAN)

Thanks them for their great works!
