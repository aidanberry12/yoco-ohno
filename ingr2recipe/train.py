# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 14:46:47 2022

@author: Rory
"""

# adapted from: https://github.com/facebookresearch/inversecooking/blob/master/src/train.py

import copy
import mlflow
import torch
from torch import nn
from  model import TransformerModel,TransformerDecoderModel
import data_loader
import args
import tqdm
import pandas as pd
import numpy as np
from build_vocab import Vocabulary



def train(model,split,criterion,optimizer,input_batch_size,dataloader,instrs_vocab_size,vocab,max_batches=None,device='cpu',total_epochs=0):    
    if max_batches==None:
        max_batches=len(dataloader)
     
    print('training: '+split)

    def _train():
        total_loss=0
        batch_idx = 0
        for targets, data in tqdm.tqdm(dataloader, desc=f'Epoch with {max_batches} batches', total=max_batches):
            #print(targets.shape)
        
            if batch_idx > max_batches:
                break

            data=torch.transpose(data,1,0).to(device) # flip so batch is second dim
            targets=torch.transpose(targets, 1, 0).to(device)
            out = model(data, targets[0:-1,:])
            
            targets = torch.transpose(targets, 1, 0).contiguous()
            out = torch.transpose(torch.transpose(out, 1, 0), 1, 2).contiguous()
            loss = criterion(out, targets[:,1:]) # Check if these dimensions are correct
            
            if split=='train':
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
            
            if device.type == 'cuda':
                loss = loss.detach().cpu().numpy()
            else:
                loss = loss.detach().numpy()

            if split == 'val' and batch_idx == max_batches-1:
                sample_idx = np.random.choice(np.arange(out.shape[0]))
                target_sentence = ' '.join([vocab.idx2word[idx] for idx in targets[sample_idx].cpu().numpy()])
                output_sentence = ' '.join([vocab.idx2word[idx] for idx in torch.argmax(out[sample_idx], dim=0).cpu().numpy()])
                mlflow.log_text(target_sentence + '\n' + output_sentence, f"sample_text_{sample_idx}_epoch_{total_epochs}.txt")
            
            total_loss += loss
            
            batch_idx += 1

        return total_loss/batch_idx
    
    if split == 'val':
        model.eval()
        with torch.no_grad():
            total_loss = _train()
    else:
        model.train()
        total_loss = _train()

    return total_loss

if __name__ == '__main__':
    
    arg=args.get_parser()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mlflow.log_param('Batch Size', arg.batch_size)
    mlflow.log_param('Max Sequence Lenth', arg.maxseqlen)
    mlflow.log_param('Max Number of Labels', arg.maxnumlabels)
    mlflow.log_param('Embedding Size', arg.embed_size)
    mlflow.log_param('Number of Heads', arg.nhead)
    mlflow.log_param('Hidden Layer Size', arg.d_hid)
    mlflow.log_param('Base Learning Rate', arg.learning_rate)
    mlflow.log_param('LR Decay Rate', arg.lr_decay_rate)
    mlflow.log_param('LR Decay Frequency', arg.lr_decay_every)
    mlflow.log_param('Number of Encoder Layers', arg.num_encoder_layers)
    mlflow.log_param('Number of Decoder Layers', arg.num_decoder_layers)
    mlflow.log_param('Dropout Rate', arg.dropout)
    mlflow.log_param('Number of Epochs', arg.num_epochs)
    
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
    model = TransformerDecoderModel(ingr_vocab_size,
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
    
    optimizer = torch.optim.Adam(model.parameters(), lr=arg.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, arg.lr_decay_every, arg.lr_decay_rate)
    
    num_batches=None
    save_every=2
    
    train_loss=[]
    val_loss=[]
    print('-' * 89)
    print('Beginning Training')
    mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
    mem = mem_params + mem_bufs # in bytes
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
                    best_model = copy.deepcopy(model)
        

            
            
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | '
                     f'valid loss {total_loss:5.2f}')
        print('-' * 89)
    torch.save(model.state_dict(), f'output/ingr2recipe_{epoch}.ckpt')
    pd.DataFrame(data={'epoch':range(epoch+1),'training_loss':train_loss,'val_loss':val_loss}).to_csv(f'output/loss_{epoch}.csv')
            
        
    
    
            
        
    
    
        
