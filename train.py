# for training with crossvalidation

import torch
import numpy as np
from dataset import getNycData
from transformers import AutoImageProcessor, Mask2FormerConfig, Mask2FormerForUniversalSegmentation
from torchgeo.samplers import RandomGeoSampler, RandomBatchGeoSampler
from torch.utils.data import DataLoader, TensorDataset, random_split, SubsetRandomSampler
# torch CUDA support
device = 'CUDA' if torch.cuda.is_available else 'cpu'

processor = AutoImageProcessor.from_pretrained('facebook/mask2former-swin-large-cityscapes-semantic')
config = Mask2FormerConfig.from_pretrained('facebook/mask2former-swin-large-cityscapes-semantic')
model = Mask2FormerForUniversalSegmentation.from_pretrained('facebook/mask2former-swin-large-cityscapes-semantic').to(device)

dataset = getNycData()
dataloader = DataLoader(dataset, batch_size=16, sampler=RandomBatchGeoSampler(dataset, batch_size=16, num_batches=100, drop_last=True))