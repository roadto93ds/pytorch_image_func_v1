# パッケージのimport
import glob
import os
import os.path as osp
import random
import numpy as np
import json
from PIL import Image
# from tqdm import tqdm
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set()
%matplotlib inline

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import models, transforms

%pip install torchinfo | tail -n 1
from torchinfo import summary

import warnings
warnings.filterwarnings('ignore')

def torch_seed(seed=0):
    """Fixed seed value."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

def filelist_to_imgs(file_list, rows=3, columns=5, scale=3, order=True):
    """
    display rows*columns images
    file_list：path_list
    scale：display scaling value
    order：path_order True/False
    """
    plt.figure(figsize=(scale*columns, scale*rows))
    if order == True:
        for i, img_path in enumerate(file_list[0:rows*columns]):
            img = Image.open(img_path)
            ax = plt.subplot(rows, columns, i+1)
            plt.imshow(img)
            # plt.axis("off")
        plt.show()

    if order == False:
        for i, img_path in enumerate(random.sample(file_list, rows*columns)):
            img = Image.open(img_path)
            ax = plt.subplot(rows, columns, i+1)
            plt.imshow(img)
            # plt.axis("off")
        plt.show()

def get_mean_std(resize, path_list):
    """
    resize：int
    path_list：only train -> train_path_list / all -> train_path_list+valid_path_list
    return：mean, std
    """

    resize = [resize,resize]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(resize)
        ])

    imgs = torch.stack([transform(Image.open(path_list).convert('RGB')) for path_list in path_list], dim=3)


    imgs_mean = imgs.view(3,-1).mean(dim=1)
    imgs_std = imgs.view(3,-1).std(dim=1)
    return imgs_mean, imgs_std

# 正規化されてtensorになった画像をmatplotlibで表示できるように再変換
def tensor2image_np(image_t, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    """
    Normalize tensor -> inverse_transform -> np.array
    Args:
        image_t: A pytorch Tensor
    Return:
        image_np
    """
    image_np = image_t.cpu().detach().numpy().transpose((1,2,0))
    mean = np.array(mean)
    std = np.array(std)
    image_np = std * image_np + mean # inverse_transform
    image_np = np.clip(image_np, 0, 1)
    return image_np

def tensor_to_img(image_t, Dim = 3):
    """
    image_t：torch.Tensor
    display one image
    tensor -> display image
    """
    if Dim == 3:
        image = image_t.cpu().detach().numpy().transpose((1,2,0)) # CHW -> HWC

    if Dim == 1:
        image = image_t[0].cpu().detach().numpy()

    image = np.clip(image, 0, 1) # 0~1clip
    plt.imshow(image, cmap="gray")
    plt.show()

def dataset_to_imgs(dataset, rows=3, columns=5, scale=3, order=True, has_label=True):
    """
    display rows*columns images
    file_list：path_list
    scale：display scaling value
    order：dataset_order True/False
    """
    # index を作る。0番目からdatasetの長さ。
    dataset_index = [i for i in range(len(dataset))]

    plt.figure(figsize=(scale*columns, scale*rows))

    if order == True:   
        for i, dataset_index in enumerate(dataset_index[0: rows*columns]):
            if has_label == True:
                img = dataset[dataset_index][0]
            if has_label == False:
                img = dataset[dataset_index]
            img = img.cpu().detach().numpy().transpose(1,2,0)
            img = np.clip(img, 0, 1) # 0~1clip
            ax = plt.subplot(rows, columns, i+1)
            plt.imshow(img)
        plt.show()

    if order == False:
        for i, dataset_index in enumerate(random.sample(dataset_index, rows*columns)):
            if has_label == True:
                img = dataset[dataset_index][0]
            if has_label == False:
                img = dataset[dataset_index]
            img = img.cpu().detach().numpy().transpose(1,2,0)
            img = np.clip(img, 0, 1) # 0~1clip
            ax = plt.subplot(rows, columns, i+1)
            plt.imshow(img)
        plt.show()

def show_label_balance(train_dataset, valid_dataset):
    """
    confirm dataset has label
    """
    train_label = [labels for imgs, labels in train_dataset]
    valid_label = [labels for imgs, labels in valid_dataset]

    train_n_class = len(set(train_label))
    valid_n_class = len(set(valid_label))
    print("train_n_class: ", train_n_class)
    print("valid_n_class: ", valid_n_class)

    plt.figure(figsize=(15,5))
    ax = plt.subplot(1, 2, 1)
    sns.countplot(train_label)
    plt.title("train_label")

    ax = plt.subplot(1, 2, 2)
    sns.countplot(valid_label)
    plt.title("valid_label")

    plt.show()

def detail_dataloader(dataset, dataloader):
    """
    show dataset -> dataloader flow
    """
    print("dataset: ",len(dataset))
    print("batch_size: ", dataloader.batch_size)
    print("dataloader: ",len(dataloader))

    data_iter = iter(dataloader)
    images, labels = next(data_iter)
    print("===== extract =====")
    print("images shape: ", images.shape)
    print("labels shape: ", labels.shape)

def dataloader_to_imgs(dataloader, net=None, device=None, has_label = True):
    """
    display dataloader_imags(max:50）
    net：classification_model
    """
    data_iter = iter(dataloader)
    if has_label == True:
        images, labels = next(data_iter)
    
    elif has_label == False:
        images = next(data_iter)

    n_size = min(len(images), 50) # n_size: len(images) or 50

    # has classification_model
    # if net is not None:
    #   inputs = images.to(device)
    #   labels = labels.to(device)

    #   # predict
    #   outputs = net(inputs)
    #   predicted = torch.max(outputs, 1)[1]

    #   images = images.to("cpu")
    #   labels = labels.to("cpu")
    #   predicted = predicted.to("cpu")

    plt.figure(figsize=(20,15))
    for i in range(n_size):
        ax = plt.subplot(5,10, i+1)
        # if net is not None:
        #   if labels[i].item() == predicted[i].item():
        #     c = "k"
        #   else:
        #     c="b"
        #   ax.set_title(str(labels[i].item()) + "/" + str(predicted[i].item()) , c=c, fontsize=20)

        # else:
            # ax.set_title(labels[i].item(), fontsize=20) # .item -> delete tensor
        image_np = images[i].numpy().copy()
        img = np.transpose(image_np, (1,2,0))
        img = np.clip(img, 0,1)

        ax.set_axis_off()
        plt.imshow(img)

    plt.show()

def transform_test(file_list, index=0 ,transform=None):
    """
    an image：file_list[index] -> transform
    display original_image and transformed_image
    """
    plt.figure(figsize=(8, 8))
    # original_image
    img = Image.open(file_list[index])
    print("original_img: ",np.array(img).shape)
    ax = plt.subplot(1, 2, 1)
    # plt.axis("off")
    plt.imshow(img)

    # transformed_image
    img_t = transform(img)
    print("transformed_img: ",img_t.shape)
    img_t = img_t.numpy().transpose((1,2,0)) # CHW -> HWC
    img_t = np.clip(img_t, 0, 1) # 0~1clip
    ax = plt.subplot(1, 2, 2)
    # plt.axis("off")
    plt.imshow(img_t)

    plt.show()

def net_test(net, dataset):
    """
    dataset -> an image -> net
    """
    img = dataset[0][0]
    print("get_image: ", img.shape)
    print("unsqueeze: ", img.unsqueeze(0).shape)
    img = img.unsqueeze(0)
    print("result: ", net(img), net(img).shape)

def Generator_test(netG, batch_size, z_dim):
    """
    given: noize = (batch_size, z_dim, 1 ,1) -> netG
    display: out[0]
    """
    sample_input = torch.randn(batch_size, z_dim, 1, 1) # 10個のz_dim次元ベクトル
    print("input_z: ", sample_input.shape) # 生成したノイズ(B,C,H,W)
    print("********** debug_forward **********")
    netG.debug_forward(sample_input) # .debug_forward 定義してるなら走る
    out = netG(sample_input)
    print("output: ", out.shape) # 出力
    tensor_to_img(out[0]) # 画像表示

def Discriminator_test(netD, dataloader):
    """
    input: dataloader
    output: 
    """
    data_iter = iter(dataloader)
    imgs = next(data_iter)
    print("input_imgs: ", imgs.shape) # dataloaderからの画像を入れる
    print("********** debug_forward **********")
    netD.debug_forward(imgs) # .debug_forward 定義してるなら走る
    out = netD(imgs)
    print("output: ", out.shape) # 出力

def show_net_status(net, dataloader):
    """
    summary input：dataloader's B,C,H,W
    """
    print("net")
    print(net)
    print("******************************************************")
    numel_list = [p.numel() for p in net.parameters()]
    print("total_params: ", sum(numel_list))
    print("list_params: ", numel_list)
    for imgs, labels in dataloader:
        B, C, H, W = imgs.shape
        break
    print("********** model summary **********")
    print(summary(net, (B, C, H, W), device="cpu"))

def show_Generator_status(netG, batch_size, z_dim):
    """
    show netG status
    given: noize = (batch_size, z_dim, 1 ,1)
    """
    print("netG")
    print(netG)
    print("******************************************************")
    numel_list = [p.numel() for p in netG.parameters()]
    print("total_params: ", sum(numel_list))
    print("list_params: ", numel_list)
    sample_input = torch.randn(batch_size, z_dim, 1, 1)
    B , C, H, W = sample_input.shape
    print("********** Generator summary **********")
    print(summary(netG, (B,C,H,W), device="cpu"))

def show_Discriminator_status(netD, dataloader):
    """
    show netD status
    summary input：dataloader's B,C,H,W
    """
    print("netD")
    print(netD)
    print("******************************************************")
    numel_list = [p.numel() for p in netD.parameters()]
    print("total_params: ", sum(numel_list))
    print("list_params: ", numel_list)
    data_iter = iter(dataloader)
    imgs = next(data_iter)
    B , C, H, W = imgs.shape
    print("********** Discriminator summary **********")
    print(summary(netD, (B,C,H,W), device="cpu"))

# 重みとバイアスの初期化（完全ランダムよりこっちの方が良い？？）
def weights_init(m):
    """
    m：model
    Conv / BatchNorm weights initialization（based normal distribution） 
    """
    classname = m.__class__.__name__
    # findは Conv という文字列が あれば何文字目か / なければ -1 を返す ので if は Conv がある場合を指定している
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02) # 正規分布
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def generate_result(G, D, z_dim, batch_size=64, device="cpu"):
    """
    noise -> Generator -> fake_images
    Return:
        fake_images
        .device = cpu
    """
    G.eval()
    D.eval()

    z = torch.randn(batch_size, z_dim).to(device)
    z = z.view(z.size(0),z.size(1),1,1)

    fake_images = G(z)
    # ret ってなんでDiscriminatorここで使う？？
    ret = D(fake_images).view(-1).to('cpu').detach().numpy() # retって何から取ってる？
    
    fake_images = fake_images.to('cpu').detach()
    return fake_images

def show_images(images,filename=None,ncols=8, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    """
    images : torch.Size([B, C, H, W]) .device=cpu
    generate_result から show_images に流せばok
    """
    nImages = images.shape[0]
    width = images.shape[3]
    height = images.shape[2]
    nrows = nImages // ncols
    
    buf = np.zeros((ncols*height, nrows*width, 3))
    idx = 0
    for r in range(nrows):
        for c in range(ncols):
            if idx >= nImages:
                continue
            buf[c*height:(c+1)*height, r*width:(r+1)*width,:] = tensor2image_np(images[idx],mean,std)
            idx += 1

    fig, ax = plt.subplots()
    ax.imshow(buf)
    if filename is None:
        plt.show()
    else:
        fig.savefig(filename)
    plt.close()



def visualize_process(train_loss_log, valid_loss_log, train_acc_log, valid_acc_log):
    """
    4params = def training 's return
    """
    plt.figure(figsize=(15,5))
    ax = plt.subplot(1, 2, 1)
    plt.plot(train_loss_log, label="train")
    plt.plot(valid_loss_log, label="valid")
    plt.title("Loss_log")
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.legend()

    ax = plt.subplot(1, 2, 2)
    plt.plot(train_acc_log, label="train")
    plt.plot(valid_acc_log, label="valid")
    plt.title("Acc_log")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()

    plt.show()
