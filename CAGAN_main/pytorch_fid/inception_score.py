import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
import time
from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy


# torch版本 但是速度极慢 49s   而且是IS_max == 9.6版本, 正常的是mean=11.31版本
# inception_score(IgnoreLabelDataset(cifar), cuda=True, batch_size=32, resize=True, splits=10)
def inception_score(imgs, cuda=True, batch_size=32, resize=True, splits=10):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]  # [3, 32, 32]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    # list -> numpy need s    numpy -> tensor need
    # 将列表转换为 numpy 数组  #数据集是 torch.Size([3, 32, 32])  # 但numpy是 imgs[0].shape = (32,32,3)
    t1 = time.time()
    imgs_np = np.array(imgs)
    t2 = time.time()
    # print("time of list to numpy change:", t2-t1)
    t1 = time.time()
    imgs_np = np.cast[np.float32]((-128 + imgs_np) / 128.)
    t2 = time.time()
    # print("time of RGB 255 to 1 change:", t2 - t1)

    # 将 numpy 数组转换为 PyTorch 张量，并将维度调整为 (N, C, H, W)
    imgs_tensor = torch.from_numpy(imgs_np.transpose((0, 3, 1, 2))) # 几毫秒
    imgs = imgs_tensor


    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)

    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i * batch_size:i * batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k + 1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


if __name__ == '__main__':
    class IgnoreLabelDataset(torch.utils.data.Dataset):
        def __init__(self, orig):
            self.orig = orig

        def __getitem__(self, index):
            return self.orig[index][0]

        def __len__(self):
            return len(self.orig)


    import torchvision.datasets as dset
    import torchvision.transforms as transforms
    import time
    from time import strftime
    from time import gmtime

    torch.cuda.set_device(1)
    start_time = time.time()

    cifar = dset.CIFAR10(root='/home/yangyeming/NASGAN/ProxeyEAGAN/datasets/cifar10', download=True,
                         transform=transforms.Compose([
                             transforms.Resize(32),
                             transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                         ])
                         )

    IgnoreLabelDataset(cifar)

    print("Calculating Inception Score...")
    print(inception_score(IgnoreLabelDataset(cifar), cuda=True, batch_size=32, resize=True, splits=10))

    end_time = time.time()
    print("run time is:", end_time - start_time)