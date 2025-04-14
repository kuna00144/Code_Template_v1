import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self, model, optimizer, loss_fn, acc_fn, device, log_dir, ckpt_dir, num_epoch):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.acc_fn = acc_fn
        self.device = device
        self.log_dir = log_dir
        self.ckpt_dir = ckpt_dir
        self.num_epoch = num_epoch

        self.writer = SummaryWriter(log_dir=self.log_dir)
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def save(self, epoch):
        torch.save({
            'model': self.model.state_dict(),
            'optim': self.optimizer.state_dict()
        }, os.path.join(self.ckpt_dir, f'model_epoch{epoch}.pth'))

    def train(self, train_loader):
        self.model.to(self.device)

        for epoch in range(1, self.num_epoch + 1):
            self.model.train()
            loss_arr, acc_arr = [], []

            for batch, data in enumerate(train_loader, 1):
                inputs = data["image"].to(self.device)
                labels = data["label"].to(self.device)

                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                acc = self.acc_fn(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_arr.append(loss.item())
                acc_arr.append(acc.item())

                print(f'TRAIN: EPOCH {epoch:04d}/{self.num_epoch:04d} | BATCH {batch:04d}/{len(train_loader)}'
                      f' | LOSS: {np.mean(loss_arr):.4f} | ACC: {np.mean(acc_arr):.4f}')

            self.writer.add_scalar('loss', np.mean(loss_arr), epoch)
            self.writer.add_scalar('acc', np.mean(acc_arr), epoch)
            self.save(epoch)

        self.writer.close()
    
    def test(self, test_loader):
        self.model.eval()
        loss_arr, acc_arr = [], []
        
        with torch.no_grad(): # test할때는 gradient x
            for batch, data in enumerate(test_loader, 1):
                inputs = data["image"].to(self.device)
                labels = data["label"].to(self.device)
                
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                acc = self.acc_fn(outputs, labels)
                
                loss_arr.append(loss.item())
                acc_arr.append(acc.item())
                
                print(f'TEST: BATCH {batch:04d}/{len(test_loader)}'
                      f' | LOSS: {np.mean(loss_arr):4f} | ACC: {np.mean(acc_arr):.4f}')
                
        print(f"\n TEST RESULT -> LOSS: {np.mean(loss_arr):.4f}, ACC: {np.mean(acc_arr):.4f}")
            