import os
import sys
from torch import nn
import copy
from torch.nn import functional as F
from argparse import ArgumentParser
from tqdm import tqdm
from tensorboardX import SummaryWriter
import pickle as pkl
import numpy as np
from glob import glob
import random
import torch
import time

sys.path.append("..")
from utils.loss_utils import ExpertCLLoos

class Exp_CL_Trainer(object):
    def __init__(
        self,
        train_loader,
        valid_loader,
        batch_size,
        num_epochs,
        log_dir,
        device,
        model,
        config,
        save_path,
        lr=1e-3,
        early_stop=1e9,
        fold=0,
        criterion_name='ExpertCLLoos'
    ):
        self.device = device
        self.config = config
        self.model = model.to(self.device)
        self.early_stop = early_stop
        self.remain_step = self.early_stop
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.save_path = save_path
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.fold=fold

        self.criterion = self.configure_criterion(criterion_name, config)
        self.optimizer = self.configure_optimizer(config)

        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.tensorwriter = SummaryWriter(log_dir)

        os.makedirs(self.save_path, exist_ok=True)

        self.best_loss = 1e9

    def train_epoch(self, epoch):
        self.model.train()
        train_iterator = tqdm(
            self.train_loader, desc="Fold {}: Epoch {}/{}".format(self.fold, epoch, self.num_epochs), leave=False
        )
        loss_epoch = 0
        for x, y, exp, lens in train_iterator:
            x = x.to(self.device)
            y = y.to(self.device)
            exp = exp.to(self.device)

            features = self.model(x, lens)
            loss = self.criterion(features, exp)
            self.model.zero_grad()
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()
            loss_epoch += loss.item()
        
        loss_epoch /= len(self.train_loader)
        print(f"Fold {self.fold}: Epoch {epoch}:")
        print(f"Train Loss: {loss_epoch:.4f}")
        self.tensorwriter.add_scalar("train_loss/epoch", loss_epoch, epoch)

        eval_loss = self.evaluate(epoch)
        if eval_loss < self.best_loss:
            self.best_loss = eval_loss
            self.best_model_path = os.path.join(self.save_path, 'best_model.pth')
            torch.save(self.model.state_dict(), self.best_model_path)
            self.remain_step = self.early_stop
        else:
            self.remain_step -= 1


    __call__ = train_epoch

    def evaluate(self, epoch):
        self.model.eval()
        eval_iterator = tqdm(
            self.valid_loader, desc="Evaluation", total=len(self.valid_loader)
        )
        loss_epoch = 0
        with torch.no_grad():
            for x, y, exp, lens in eval_iterator:
                x = x.to(self.device)
                y = y.to(self.device)
                exp = exp.to(self.device)

                features = self.model(x, lens)
                loss = self.criterion(features, exp)
                loss_epoch += loss.item()
        loss_epoch /= len(self.train_loader)
        print(f"Epoch {epoch}:")
        print(f"Eval Loss: {loss_epoch:.4f} | Best Eval Loss: {self.best_loss:.4f}")
        self.tensorwriter.add_scalar("eval_loss/epoch", loss_epoch, epoch)

        return loss_epoch


    def configure_optimizer(self, config):
        return torch.optim.Adam(self.model.parameters(), lr=config.lr, betas=(config.beta1, config.beta2), weight_decay=3e-4)

    def configure_criterion(self, criterion_name, config):
        if criterion_name == 'ExpertCLLoos':
            return ExpertCLLoos(self.device)
        else:
            raise ValueError("Invalid Loss Type!")