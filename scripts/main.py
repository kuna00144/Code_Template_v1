import os
import numpy as np
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.layers import resnet18
from data.dataset import CustomDataset
from scripts.train import Trainer

lr = 0.0001
batch_size = 64
num_epoch = 10
ckpt_dir = './checkpoint'
log_dir = './log'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    #transforms.RandomCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

train_data = CustomDataset(root_dir="/home/guna/datasets/MNIST/train", transform= transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_data = CustomDataset(root_dir="/home/guna/datasets/MNIST/test", transform=transform)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

#model = VGG16().to(device)
model = resnet18().to(device)
optim = torch.optim.Adam(model.parameters(), lr=lr)
fn_loss = nn.CrossEntropyLoss().to(device)

def fn_acc(output, label):
    pred = torch.softmax(output, dim=1).argmax(dim=1)
    return (pred == label).float().mean()

trainer = Trainer(
    model = model,
    optimizer = optim,
    loss_fn = fn_loss,
    acc_fn = fn_acc,
    device = device,
    log_dir = log_dir,
    ckpt_dir = ckpt_dir,
    num_epoch = num_epoch
)

trainer.train(train_loader)

trainer.test(test_loader)