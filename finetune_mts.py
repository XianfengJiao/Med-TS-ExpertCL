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
from trainers import CL_Trainer, UTS_Trainer, MTS_Trainer
from datasets import Aug_Dataset, UTS_Dataset, MTS_Dataset
from models import GRU_encoder, MC_GRU_predictor

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
    # ------------------------ fix random seeds for reproducibility ------------------------
    setup_seed(args.seed)
    
    # ------------------------ Load Data ------------------------
    x, y = load_data(args.x_path, args.y_path, args)

    # ------------------------ kfold ------------------------
    kfold = KFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
    kfold_metrics = {}
    for i, (train_set, val_set) in enumerate(kfold.split(x, y)):
        train_dataset = MTS_Dataset(
            input_data=np.array(x, dtype=object)[train_set],
            input_label=np.array(y, dtype=object)[train_set],
            config=config,
            kfold=i,
        )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)

        valid_dataset = MTS_Dataset(
            input_data=np.array(x, dtype=object)[val_set],
            input_label=np.array(y, dtype=object)[val_set],
            config=config,
            kfold=i,
        )
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=valid_dataset.collate_fn)

        # ----------------- Instantiate Model --------------------------
        model = load_model(args.model_name, config)

        # ----------------- Load Checkpoint --------------------------
        if config.pretrian_ckpt_path is not None:
            
            all_f_dict = {}
            for f_i, f_name in enumerate(config.feature_names):
                ckpt_path = config.pretrian_ckpt_path.replace('xxx', f_name)
                ckpt_path = os.path.join(ckpt_path, 'kfold-'+str(i), 'best_model.pth')
                f_dict = torch.load(ckpt_path)
                f_dict = {k.replace('gru', 'GRUs.'+str(f_i)): v for k, v in f_dict.items()}
                all_f_dict.update(f_dict)

            load_info = model.load_state_dict(all_f_dict, strict=False)
            if config.freeze:
                for p in model.GRUs.parameters():
                    p.requires_grad=False

            print('Load checkpoint from', ckpt_path)
            print('missing_keys:', load_info.missing_keys)
        
        # ----------------- Instantiate Trainer ------------------------
        trainer = MTS_Trainer(
            train_loader=train_loader,
            valid_loader=valid_loader,
            batch_size=args.batch_size,
            num_epochs=args.epoch,
            fold=i,
            log_dir=os.path.join(args.log_dir, 'kfold-'+str(i)),
            lr=args.lr,
            device=device,
            config=config,
            monitor=args.monitor,
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
        copyfile(trainer.best_model_path, trainer.best_model_path.replace('.pth','_' + trainer.monitor + '_' + str(trainer.best_metric) + '.pth'))

        # ----------------- Save Metrics -------------------------
        for key, value in trainer.metric_all.items():
            if key not in kfold_metrics:
                kfold_metrics[key] = [value]
            else:
                kfold_metrics[key].append(value)
    
    # ----------------- Print Metrics ------------------------
    print('#'*20,"End testing in all fold",'#'*20)
    for key, value in kfold_metrics.items():
        print('%s: %.4f(%.4f)'%(key, np.mean(value), np.std(value)))


    print(f"Training time is : {datetime.now()-start_time}")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--res_path', default='/home/v-xiajiao/code/Med-TS-ExpertCL/', type=str)
    parser.add_argument('--ckpt_save_path', default='/home/v-xiajiao/code/Med-TS-ExpertCL/checkpoints/finetune/debug_mts', type=str)
    parser.add_argument('--config_path', default='/home/v-xiajiao/code/Med-TS-ExpertCL/configs/pretrain_cl_gru_encoder.yaml', type=str)
    parser.add_argument('--log_dir', default='/home/v-xiajiao/code/Med-TS-ExpertCL/logs/finetune/debug_mts', type=str)
    parser.add_argument('--pretrian_ckpt_path', default=None, type=str)
    parser.add_argument('--model_name', default='MC_GRU_BN_predictor', type=str)
    parser.add_argument('--criterion_name', default='mse', type=str)
    parser.add_argument('--monitor', default='auroc', type=str)
    parser.add_argument('--x_path', default='/home/v-xiajiao/code/Med-TS-ExpertCL/input/Sepsis/x.pkl', type=str)
    parser.add_argument('--y_path', default='/home/v-xiajiao/code/Med-TS-ExpertCL/input/Sepsis/y.pkl', type=str)
    parser.add_argument('--feature_names', default='hr,o2sat,Temp,sbp,map,dbp,Resp,etco2,baseexcess,hco3,fio2,ph,paco2,sao2,ast,bun,Alkalinephos,Calcium,Chloride,creatinine_creatinine,Bilirubin_direct,Glucose,Lactate,Magnesium,Phosphate,Potassium,Bilirubin_total,troponini,Hct,Hgb,ptt,wbc,Fibrinogen,Platelets', type=str)
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--seed', default=5, type=int)
    parser.add_argument('--kfold', default=5, type=int)
    parser.add_argument('--n2n', action='store_true')
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--early_stop', default=20, type=int)
    parser.add_argument('--hidden_dim', default=64, type=int)
    args = parser.parse_args()

    args.feature_names = [n for n in args.feature_names.split(',')]

    main(args)