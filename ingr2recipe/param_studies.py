# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 13:15:06 2022

@author: Rory
"""
import optuna
from optuna.pruners import SuccessiveHalvingPruner
import copy
import mlflow
import torch
from torch import nn
from  model import TransformerModel
import data_loader
import args
import tqdm
import pandas as pd
import numpy as np
from build_vocab import Vocabulary
from train import train
import joblib

def objective(trial):
    arg=args.get_parser()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    learning_rate=trial.suggest_float("learning_rate",0.001,0.1,log=True)
    lr_decay_rate=trial.suggest_float("lr_decay_rate",0.8,0.999,log=True)
    lr_decay_every=trial.suggest_int('lr_decay_every',1,16,step=5)    

    # Build data loader
    data_loaders = {}
    datasets = {}
    print('Device: ' + str(device))
    for split in ['train', 'val']:
        max_num_samples = max(arg.max_eval, arg.batch_size) if split == 'val' else -1
        max_num_samples = max(arg.max_eval, arg.batch_size) if split == 'val' else -1
        # num_samples=arg.training_samples if split=='train' else arg.validation_samples 
        print('Loading data for '+split)
        data_loaders[split], datasets[split] = data_loader.get_loader(arg.recipe1m_dir, arg.aux_data_dir, split,
                                                              arg.maxseqlen,
                                                              arg.maxnuminstrs,
                                                              arg.maxnumlabels,
                                                              arg.maxnumims,
                                                              None, 
                                                              arg.batch_size,
                                                              shuffle=split == 'train', 
                                                              num_workers=arg.num_workers,
                                                              drop_last=True,
                                                              max_num_samples=max_num_samples,
                                                              use_lmdb=arg.use_lmdb,
                                                              suff=arg.suff,
                                                              use_small_data=arg.use_small_data
                                                              )
        for data in data_loaders[split]:
            for item in data:
                item = item.to(device) # send to cuda 
    
    print('Dataloader created')
    # Get size of ingredient vocab and instruction vocab  
    ingr_vocab_size = datasets[split].get_ingrs_vocab_size()
    instrs_vocab_size = datasets[split].get_instrs_vocab_size()
    print('-' * 89)
    print('Creating Model')
    # Create Transformer model
    model = TransformerModel(ingr_vocab_size,
                             instrs_vocab_size, 
                             arg.embed_size,
                             arg.nhead,
                             arg.d_hid, 
                             arg.num_encoder_layers,
                             arg.num_decoder_layers,
                             arg.dropout,
                             arg.maxnumlabels,
                             arg.maxnuminstrs*arg.maxseqlen,
                             device).to(device)
    print('Done Creating Model')
    print('-' * 89)
    criterion = nn.CrossEntropyLoss(ignore_index=23030).to(device)
    best_val_loss = float('inf')
    best_model = None
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,lr_decay_every, lr_decay_rate)
    
    num_batches=None
    
    train_loss=[]
    val_loss=[]
    print('-' * 89)
    print('Beginning Training')

    for epoch in range(arg.num_epochs):
        print('starting epoch '+str(epoch))
        for split in ['train','val']:
            loader = data_loaders[split]
            total_loss = train(model,split,criterion,optimizer,arg.batch_size,loader,instrs_vocab_size,datasets[split].get_instrs_vocab(),max_batches=num_batches,device=device,total_epochs=epoch)

            mlflow.log_metric(f'{split} loss',total_loss,epoch)
            
            if split=='train':
                train_loss.append(total_loss)
                scheduler.step()
            else:
                val_loss.append(total_loss)
                if total_loss < best_val_loss:
                    best_val_loss = total_loss
                    # best_model = copy.deepcopy(model)  
                    
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | '
                     f'valid loss {total_loss:5.2f}')
        print('-' * 89)

    return total_loss
    
    

if __name__ == '__main__':
    study = optuna.create_study(direction = "minimize",pruner=optuna.pruners.SuccessiveHalvingPruner())
    study.optimize(objective, n_trials = 20)
    trial = study.best_trial
    print("Best Score: ", trial.value)
    print("Best Params: ")
    for key, value in trial.params.items():
        print("  {}: {}".format(key, value))
    joblib.dump(study, "study.pkl")
    
    