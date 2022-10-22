import torch
import pickle as pkl
import os
from omegaconf import OmegaConf
import numpy as np
from shutil import copyfile
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from datetime import datetime
from argparse import ArgumentParser
from utils.schedule_utils import setup_seed, load_config_file, load_model
from sklearn.model_selection import StratifiedKFold, KFold
from trainers import CL_Trainer
from datasets import Aug_Dataset
from models import GRU_encoder

start_time = datetime.now()

def load_data(x_path, y_path, config):
    x = pkl.load(open(x_path, 'rb'))
    y = pkl.load(open(y_path, 'rb'))
    if not config.n2n:
        y = [yy[-1] for yy in y]
    return x, y

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # ------------------------ load config ------------------------
    config = load_config_file(args.config_path)
    config = OmegaConf.merge(config, vars(args))
    OmegaConf.save(config=config, f=os.path.join(config.log_dir, 'config.yml'))
    # ------------------------ fix random seeds for reproducibility ------------------------
    setup_seed(args.seed)
    
    # ------------------------ Load Data ------------------------
    x, y = load_data(args.x_path, args.y_path, args)

    # ------------------------ kfold ------------------------
    kfold_loss = []
    kfold = KFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
    for i, (train_set, val_set) in enumerate(kfold.split(x, y)):
        # ------------------------ Instantiate Dataset ------------------------
        train_dataset = Aug_Dataset(
            input_data=np.array(x, dtype=object)[train_set],
            config=config,
            kfold=i,
            aug_type=args.aug_type
        )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)

        valid_dataset = Aug_Dataset(
            input_data=np.array(x, dtype=object)[val_set],
            config=config,
            kfold=i,
            aug_type=args.aug_type
        )
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=valid_dataset.collate_fn)

        # ----------------- Instantiate Model --------------------------
        model = load_model(args.model_name, config)
        
        # ----------------- Instantiate Trainer ------------------------
        trainer = CL_Trainer(
            train_loader=train_loader,
            valid_loader=valid_loader,
            batch_size=args.batch_size,
            num_epochs=args.epoch,
            fold=i,
            log_dir=os.path.join(args.log_dir, 'kfold-'+str(i)),
            lr=args.lr,
            device=device,
            config=config,
            early_stop=args.early_stop,
            model=model,
            save_path=os.path.join(args.ckpt_save_path, 'kfold-'+str(i)),
            criterion_name=args.criterion_name
        )

        # ----------------- Start Training ------------------------
        print('#'*20,"kfold-{}: starting training...".format(i),'#'*20)
        for epoch in range(1, args.epoch + 1):
            trainer.train_epoch(epoch)
            if trainer.remain_step == 0:
                break
        print('#'*20,"kfold-{}: end training...".format(i),'#'*20)
        print('Eval best loss:', trainer.best_loss)
        kfold_loss.append(trainer.best_loss)
        copyfile(trainer.best_model_path, trainer.best_model_path.replace('.pth', '_'+str(trainer.best_loss)+'.pth'))

    print('#'*20,"End testing in all fold",'#'*20)
    print("Loss: {:.4f} ({:.4f})".format(np.mean(kfold_loss), np.std(kfold_loss)))
    print(f"Training time is : {datetime.now()-start_time}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--ckpt_save_path', default='/home/v-xiajiao/code/Med-TS-ExpertCL/checkpoints/pretrian/debug_alb_week', type=str)
    parser.add_argument('--log_dir', default='/home/v-xiajiao/code/Med-TS-ExpertCL/logs/pretrian/debug_alb_week', type=str)
    parser.add_argument('--config_path', default='/home/v-xiajiao/code/Med-TS-ExpertCL/configs/pretrain_cl_gru_encoder.yaml', type=str)
    parser.add_argument('--model_name', default='GRU_encoder', type=str)
    parser.add_argument('--criterion_name', default='NTXentLoss', type=str)
    parser.add_argument('--aug_type', default='week', type=str)
    parser.add_argument('--x_path', default='/home/v-xiajiao/code/Med-TS-ExpertCL/input/CKD/alb_demo_data/alb_x.pkl', type=str)
    parser.add_argument('--y_path', default='/home/v-xiajiao/code/Med-TS-ExpertCL/input/CKD/y.pkl', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--seed', default=5, type=int)
    parser.add_argument('--kfold', default=5, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--n2n', action='store_true')
    parser.add_argument('--merge', action='store_true')
    parser.add_argument('--early_stop', default=20, type=int)
    parser.add_argument('--hidden_dim', default=64, type=int)
    args = parser.parse_args()
    main(args)