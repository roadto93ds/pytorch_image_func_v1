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
import seaborn as sns
sns.set()
# %matplotlib inline

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import models, transforms

# !pip install torchinfo | tail -n 1
from torchinfo import summary

import warnings
warnings.filterwarnings('ignore')

# 乱数の固定化
def torch_seed(seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

def filelist_to_imgs(file_list, rows=3, columns=5, scale=3, order=True):
  """
  rows*columns枚の画像を表示
  file_list:パスのリスト
  rows：行数
  columns:列数
  scale：調節用
  order：True：前から順番通り / False：ランダム抽出
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
  resize = [resize,resize]

  transform = transforms.Compose([
                  transforms.ToTensor(),  # テンソルに変換
                  transforms.Resize(resize)
                  ])

  imgs = torch.stack([transform(Image.open(path_list)) for path_list in path_list], dim=3)

  imgs_mean = imgs.view(3,-1).mean(dim=1)
  imgs_std = imgs.view(3,-1).std(dim=1)
  
  # print("stock: ", imgs.shape)
  # print("mean: ",imgs_mean)
  # print("std: ", imgs_std)

  return imgs_mean, imgs_std

def dataset_to_imgs(dataset, rows=3, columns=5, scale=3, order=True):
  """
  rows*columns枚の画像を表示
  file_list:パスのリスト
  rows：行数
  columns:列数
  scale：調節用
  order：True：前から順番通り / False：ランダム抽出
  """
  # index を作る。0番目からdatasetの長さ。
  dataset_index = [i for i in range(len(dataset))]

  plt.figure(figsize=(scale*columns, scale*rows))
  if order == True:   
    for i, dataset_index in enumerate(dataset_index[0: rows*columns]):
      img = dataset[dataset_index][0]
      img = img.cpu().detach().numpy().transpose(1,2,0)
      img = np.clip(img, 0, 1) # 0~1clip
      ax = plt.subplot(rows, columns, i+1)
      plt.imshow(img)
    plt.show()

  if order == False:
    for i, dataset_index in enumerate(random.sample(dataset_index, rows*columns)):
      img = dataset[dataset_index][0]
      img = img.cpu().detach().numpy().transpose(1,2,0)
      img = np.clip(img, 0, 1) # 0~1clip
      ax = plt.subplot(rows, columns, i+1)
      plt.imshow(img)
    plt.show()

def show_label_balance(train_dataset, valid_dataset):
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
  print("dataset: ",len(dataset))
  print("batch_size: ", dataloader.batch_size)
  print("dataloader: ",len(dataloader))

  for images, labels in dataloader:
    break

  print(images.shape)
  print(labels.shape)

def dataloader_to_imgs(dataloader, net=None, device=None):
  """
  最初のデータローダーのセットから画像を表示
  最大表示枚数は50枚
  classificationを行うモデルを引き渡して予測ラベルも合わせて表示できる
  """
  for images, labels in dataloader:
    break

  n_size = min(len(images), 50) # 最大でも50枚制限

  if net is not None:
    inputs = images.to(device)
    labels = labels.to(device)

    # 予測計算
    outputs = net(inputs)
    predicted = torch.max(outputs, 1)[1]

    images = images.to("cpu")
    labels = labels.to("cpu")
    predicted = predicted.to("cpu")

  plt.figure(figsize=(20,15))
  for i in range(n_size):
    ax = plt.subplot(5,10, i+1)
    if net is not None:
      if labels[i].item() == predicted[i].item():
        c = "k"
      else:
        c="b"
      ax.set_title(str(labels[i].item()) + "/" + str(predicted[i].item()) , c=c, fontsize=20)
      # ax.set_title(str(labels[i].item()) + "/" , c=c, fontsize=20)

    else:
      ax.set_title(labels[i].item(), fontsize=20) # .itemでtensor表示を消す
    image_np = images[i].numpy().copy()
    img = np.transpose(image_np, (1,2,0))
    img = np.clip(img, 0,1)

    ax.set_axis_off()
    plt.imshow(img)

  plt.show()

def transform_test(file_list, index=0 ,transform=None):
  plt.figure(figsize=(8, 8))
  # original_image
  img = Image.open(file_list[index])
  print("original_img: ",np.array(img).shape)
  ax = plt.subplot(1, 2, 1)
  plt.imshow(img)

  # transformed_img
  img_t = transform(img)
  print("transformed_img: ",img_t.shape)
  img_t = img_t.numpy().transpose((1,2,0)) # CHW -> HWC
  img_t = np.clip(img_t, 0, 1) # 0~1clip
  ax = plt.subplot(1, 2, 2)
  plt.imshow(img_t)

  plt.show()

def net_test(net, dataset):
  """
  classification用
  datasetから回収した画像1枚に対して出力のテストを行う
  """
  img = dataset[0][0]
  print("get_image: ", img.shape)
  print("unsqueeze: ", img.unsqueeze(0).shape)
  img = img.unsqueeze(0)
  print("result: ", net(img), net(img).shape)

def show_net_status(net, dataloader):
  """
  パラメータの総数
  summaryは入力次元を引数に取るので、dataloaderからB,C,H,Wの次元を回収
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

# 重みとバイアスの初期化（完全ランダムよりこっちの方が良い？？）
def weights_init(m):
  # m は model
  classname = m.__class__.__name__
  # findは Conv という文字列が あれば何文字目か / なければ -1 を返す ので if は Conv がある場合を指定している
  if classname.find("Conv") != -1:
    nn.init.normal_(m.weight.data, 0.0, 0.02) # 正規分布を使って初期化
  
  elif classname.find("BatchNorm") != -1:
    nn.init.normal_(m.weight.data, 1.0, 0.02)
    nn.init.constant_(m.bias.data, 0)

def visualize_process(train_loss_log, valid_loss_log, train_acc_log, valid_acc_log):
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
