import torch
import gzip
import pickle
import numpy as np
from nets.facenet import Facenet
from torch.utils.data import DataLoader
from utils.dataloader import FacenetDataset, LFWDataset, dataset_collate

cuda            = torch.cuda.is_available()
backbone        = "mobilenet"
#----------------------------------------------------------------------------------------------------------------------------#
#   是否使用主干网络的预训练权重，此处使用的是主干的权重，因此是在模型构建的时候进行加载的。
#   如果设置了model_path，则主干的权值无需加载，pretrained的值无意义。
#   如果不设置model_path，pretrained = True，此时仅加载主干开始训练。
#   如果不设置model_path，pretrained = False，此时从0开始训练。
#----------------------------------------------------------------------------------------------------------------------------#
pretrained      = False
input_shape     = [160, 160, 3]

lfw_dir_path    = "/mnt/d/dataset/lfw-pairs/lfw_funneled"
lfw_pairs_path  = "/mnt/d/dataset/lfw-pairs/pairs.txt"

# 分类列表
with gzip.open('classes.gz', 'rb') as f:
    classes = pickle.load(f)
# 预制菜, 为了保证每次打断再train都是用的相同的训练集和验证集
with gzip.open('train-index.gz', 'rb') as f:
    index_train = pickle.load(f)
with gzip.open('val-index.gz', 'rb') as f:
    index_val = pickle.load(f)

num_classes = len(classes)

model = Facenet(backbone=backbone, num_classes=num_classes, pretrained=pretrained)

LFW_loader = torch.utils.data.DataLoader(
    LFWDataset(dir=lfw_dir_path, pairs_path=lfw_pairs_path, image_size=input_shape), batch_size=32, shuffle=False)