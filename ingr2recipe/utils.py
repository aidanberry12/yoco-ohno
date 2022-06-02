# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 20:00:20 2022

@author: Rory
"""

import args
import torch
import tqdm
from build_vocab import Vocabulary
import data_loader
import pickle
import os
from model import TransformerModel,TransformerDecoderModel

def idx2word_seq(vocab,idx_list):
    sequence=[]
    idx2word=vocab.idx2word
    for idx in idx_list:
        sequence.append(idx2word[idx.item()])
    
    return(sequence)

def ingr2recipe_test(model,args):
    max_num_samples = max(arg.max_eval, arg.batch_size)
    max_num_samples = max(arg.max_eval, arg.batch_size)
    max_batches=10
    a_dataloader, dataset = data_loader.get_loader(arg.recipe1m_dir, arg.aux_data_dir, 'val',
                                                              arg.maxseqlen,
                                                              arg.maxnuminstrs,
                                                              arg.maxnumlabels,
                                                              arg.maxnumims,
                                                              None, 
                                                              arg.batch_size,
                                                              shuffle='val' == 'train', 
                                                              num_workers=arg.num_workers,
                                                              drop_last=True,
                                                              max_num_samples=max_num_samples,
                                                              use_lmdb=arg.use_lmdb,
                                                              suff=arg.suff,
                                                              use_small_data=arg.use_small_data
                                                              )   

    instrs_vocab = pickle.load(open(os.path.join(arg.aux_data_dir, arg.suff + 'recipe1m_vocab_toks.pkl'), 'rb'))
    ingr_vocab = pickle.load(open(os.path.join(arg.aux_data_dir, arg.suff + 'recipe1m_vocab_ingrs.pkl'), 'rb'))

    target,data=iter(a_dataloader).next()
    data=torch.transpose(data,1,0)
    recipes=model.get_inference(data,0,1)
    for i in range(recipes.size()[1]):
        print(f'sample {i}')
        print('Ingredients: ')
        ingredients=idx2word_seq(ingr_vocab,data[:,i])
        for an_ingredient in ingredients:
            print(an_ingredient[0])
        print("Recipe:")
        print(idx2word_seq(instrs_vocab,recipes[:,i]))
        print('-' * 89)
    


if __name__ == '__main__':
    arg=args.get_parser()
    max_num_samples = max(arg.max_eval, arg.batch_size)
    max_num_samples = max(arg.max_eval, arg.batch_size)
    a_dataloader, dataset = data_loader.get_loader(arg.recipe1m_dir, arg.aux_data_dir, 'val',
                                                              arg.maxseqlen,
                                                              arg.maxnuminstrs,
                                                              arg.maxnumlabels,
                                                              arg.maxnumims,
                                                              None, 
                                                              arg.batch_size,
                                                              shuffle='val' == 'train', 
                                                              num_workers=arg.num_workers,
                                                              drop_last=True,
                                                              max_num_samples=max_num_samples,
                                                              use_lmdb=arg.use_lmdb,
                                                              suff=arg.suff,
                                                              use_small_data=arg.use_small_data
                                                              )   

    ingr_vocab_size = dataset.get_ingrs_vocab_size()
    instrs_vocab_size = dataset.get_instrs_vocab_size()
    checkpoint = torch.load('output//ingr2recipe_dec_only.ckpt',map_location=torch.device('cpu'))
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
                             device='cpu')
    model.load_state_dict(checkpoint)
    ingr2recipe_test(model,arg)
