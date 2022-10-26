import os
import sys
from torch import nn
import copy
from torch.nn import functional as F
from utils.metric_utils import print_metrics_binary
from tqdm import tqdm
from tensorboardX import SummaryWriter
import pickle as pkl
import numpy as np
from glob import glob
import random
import torch
import time

class MTS_Trainer(object):
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
        monitor='auprc',
        lr=1e-3,
        early_stop=1e9,
        fold=0,
        criterion_name='MSE'
    ):
        self.device = device
        self.monitor = monitor
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
        self.fold = fold

        self.criterion = self.configure_criterion(criterion_name, config)
        self.optimizer = self.configure_optimizer(config)

        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.tensorwriter = SummaryWriter(log_dir)

        os.makedirs(self.save_path, exist_ok=True)

        self.best_loss = 1e9
        self.best_metric = -1e9

    def train_epoch(self, epoch):
        self.model.train()
        train_iterator = tqdm(
            self.train_loader, desc="Fold {}: Epoch {}/{}".format(self.fold, epoch, self.num_epochs), leave=False
        )
        loss_epoch = 0
        for x, y, lens in train_iterator:
            x = x.to(self.device)
            y = y.to(self.device)
            pred = self.model(x, lens)
            # clip pred
            pred = torch.clamp(pred, min=1e-7, max=1-1e-7)
            loss = self.criterion(pred, y)
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

        eval_loss, eval_metric = self.evaluate(epoch)

        if eval_metric[self.monitor] > self.best_metric:
            self.best_metric = eval_metric[self.monitor]
            self.best_model_path = os.path.join(self.save_path, 'best_model.pth')
            torch.save(self.model.state_dict(), self.best_model_path)
            self.metric_all = eval_metric
            self.remain_step = self.early_stop
        else:
            self.remain_step -= 1

        print('-'*100)
        print('Epoch {}, best eval {}: {}'.format(epoch, self.monitor, self.best_metric))
        print('-'*100)
        
    __call__ = train_epoch

    def evaluate(self, epoch):
        self.model.eval()
        eval_iterator = tqdm(
            self.valid_loader, desc="Evaluation", total=len(self.valid_loader)
        )
        all_y = []
        all_pred = []

        with torch.no_grad():
            for x, y, lens in eval_iterator:
                x = x.to(self.device)
                y = y.to(self.device)
                
                pred = self.model(x, lens)
                
                all_y.append(y)
                all_pred.append(pred)
        
        all_y = torch.cat(all_y, dim=0).squeeze()
        all_pred = torch.cat(all_pred, dim=0).squeeze()
        
        loss = self.criterion(all_pred, all_y)
        print(f"Epoch {epoch}:")
        print(f"Eval Loss: {loss:.4f}")

        metrics = print_metrics_binary(all_pred.cpu().detach().numpy().flatten(), all_y.cpu().detach().numpy().flatten())
        self.tensorwriter.add_scalar("eval_loss/epoch", loss, epoch)
        self.tensorwriter.add_scalar("eval_auprc/epoch", metrics['auprc'], epoch)
        self.tensorwriter.add_scalar("eval_minpse/epoch", metrics['minpse'], epoch)
        self.tensorwriter.add_scalar("eval_auroc/epoch", metrics['auroc'], epoch)
        self.tensorwriter.add_scalar("eval_prec0/epoch", metrics['prec0'], epoch)
        self.tensorwriter.add_scalar("eval_acc/epoch", metrics['acc'], epoch)
        self.tensorwriter.add_scalar("eval_prec1/epoch", metrics['prec1'], epoch)
        self.tensorwriter.add_scalar("eval_rec0/epoch", metrics['rec0'], epoch)
        self.tensorwriter.add_scalar("eval_rec1/epoch", metrics['rec1'], epoch)
        self.tensorwriter.add_scalar("eval_f1_score/epoch", metrics['f1_score'], epoch)

        return loss, metrics


    def configure_optimizer(self, config):
        # return torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=config.lr, betas=(config.beta1, config.beta2), weight_decay=3e-4)

    def configure_criterion(self, criterion_name, config):
        if criterion_name == 'mse':
            return nn.MSELoss()
        elif criterion_name == 'bce':
            return nn.BCELoss()
        else:
            raise ValueError("Invalid Loss Type!")
