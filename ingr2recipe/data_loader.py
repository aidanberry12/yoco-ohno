import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
# from PIL import Image
from build_vocab import Vocabulary
import random
import json
import lmdb

from args import get_parser
import torchvision.transforms as transforms


class Recipe1MDataset(data.Dataset):

    def __init__(self, data_dir, aux_data_dir, split, maxseqlen, maxnuminstrs, maxnumlabels, maxnumims,
                 transform=None, max_num_samples=-1, use_lmdb=False, suff='',use_small_data=True):

        self.ingrs_vocab = pickle.load(open(os.path.join(aux_data_dir, suff + 'recipe1m_vocab_ingrs.pkl'), 'rb'))
        self.instrs_vocab = pickle.load(open(os.path.join(aux_data_dir, suff + 'recipe1m_vocab_toks.pkl'), 'rb'))
        
        if use_small_data:
            self.dataset = pickle.load(open(os.path.join(aux_data_dir, suff + 'recipe1m_small_'+split+'.pkl'), 'rb'))
        else:
            self.dataset = pickle.load(open(os.path.join(aux_data_dir, suff + 'recipe1m_'+split+'.pkl'), 'rb'))
        # if num_samples is not None:
        #     self.dataset=self.dataset[0:num_samples]
            
        self.label2word = self.get_ingrs_vocab()
        self.ids = []
        self.split = split
        for i, entry in enumerate(self.dataset):
            if len(entry['images']) == 0:
                continue
            self.ids.append(i)


        self.transform = transform
        self.max_num_labels = maxnumlabels
        self.maxseqlen = maxseqlen
        self.max_num_instrs = maxnuminstrs
        self.maxseqlen = maxseqlen*maxnuminstrs
        self.maxnumims = maxnumims
        # if max_num_samples != -1:
        #     random.shuffle(self.ids)
        #     self.ids = self.ids[:max_num_samples]

    def get_instrs_vocab(self):
        return self.instrs_vocab

    def get_instrs_vocab_size(self):
        return len(self.instrs_vocab)

    def get_ingrs_vocab(self):
        return [min(w, key=len) if not isinstance(w, str) else w for w in
                self.ingrs_vocab.idx2word.values()]  # includes 'pad' ingredient

    def get_ingrs_vocab_size(self):
        return len(self.ingrs_vocab)

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""

        sample = self.dataset[self.ids[index]]
        # img_id = sample['id']
        captions = sample['tokenized']
        # paths = sample['images'][0:self.maxnumims]

        idx = index

        labels = self.dataset[self.ids[idx]]['ingredients']
        title = sample['title']

        tokens = []
        tokens.extend(title)
        # add fake token to separate title from recipe
        tokens.append('<eoi>')
        for c in captions:
            tokens.extend(c)
            tokens.append('<eoi>')

        ilabels_gt = np.ones(self.max_num_labels) * self.ingrs_vocab('<pad>')
        pos = 0

        true_ingr_idxs = []
        for i in range(len(labels)):
            true_ingr_idxs.append(self.ingrs_vocab(labels[i]))

        for i in range(self.max_num_labels):
            if i >= len(labels):
                label = '<pad>'
            else:
                label = labels[i]
            label_idx = self.ingrs_vocab(label)
            if label_idx not in ilabels_gt:
                ilabels_gt[pos] = label_idx
                pos += 1

        ilabels_gt[pos] = self.ingrs_vocab('<end>')
        ingrs_gt = torch.from_numpy(ilabels_gt).long()

        caption = []

        caption = self.caption_to_idxs(tokens, caption)
        caption.append(self.instrs_vocab('<end>'))

        caption = caption[0:self.maxseqlen]
        target = torch.Tensor(caption)

        return target, ingrs_gt, self.instrs_vocab('<pad>')

    def __len__(self):
        return len(self.ids)

    def caption_to_idxs(self, tokens, caption):

        caption.append(self.instrs_vocab('<start>'))
        for token in tokens:
            caption.append(self.instrs_vocab(token))
        return caption


def collate_fn(data):

    # Sort a data list by caption length (descending order).
    # data.sort(key=lambda x: len(x[2]), reverse=True)
    captions, ingrs_gt, pad_value = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).

    # image_input = torch.stack(image_input, 0)
    ingrs_gt = torch.stack(ingrs_gt, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.ones(len(captions), max(lengths)).long()*pad_value[0]

    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return targets, ingrs_gt


def get_loader(data_dir, aux_data_dir, split, maxseqlen,
               maxnuminstrs, maxnumlabels, maxnumims, transform, batch_size,
               shuffle, num_workers, drop_last=False,
               max_num_samples=-1,
               use_lmdb=False,
               suff='',
               use_small_data=True):

    dataset = Recipe1MDataset(data_dir=data_dir, aux_data_dir=aux_data_dir, split=split,
                              maxseqlen=maxseqlen, maxnumlabels=maxnumlabels, maxnuminstrs=maxnuminstrs,
                              maxnumims=maxnumims,
                              transform=transform,
                              max_num_samples=max_num_samples,
                              use_lmdb=use_lmdb,
                              suff=suff,
                              use_small_data=use_small_data)

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                              drop_last=drop_last, collate_fn=collate_fn, pin_memory=True)
    return data_loader, dataset

def make_small_datasets(split,arg):
    dataset = pickle.load(open(os.path.join(arg.aux_data_dir, arg.suff + 'recipe1m_'+split+'.pkl'), 'rb'))
    num_samples=arg.training_samples if split=='train' else arg.validation_samples 
    dataset=dataset[0:num_samples]
    with open(os.path.join(arg.aux_data_dir, arg.suff+'recipe1m_small_' + split + '.pkl'), 'wb') as f:
        pickle.dump(dataset, f)
    

if __name__ == '__main__':
    # This is just used for testing and looking at files. This file wont typically be called as the main file
    # Build data loader
    data_loaders = {}
    datasets = {}
    
    split = 'val'
    arg=get_parser()
    # make_small_datasets('train',arg)
    make_small_datasets('val',arg)
    # max_num_samples = max(arg.max_eval, arg.batch_size) if split == 'val' else -1
    # max_num_samples = max(arg.max_eval, arg.batch_size) if split == 'val' else -1
    # data_loaders[split], datasets[split] = get_loader(arg.recipe1m_dir, arg.aux_data_dir, split,
    #                                                       arg.maxseqlen,
    #                                                       arg.maxnuminstrs,
    #                                                       arg.maxnumlabels,
    #                                                       arg.maxnumims,
    #                                                       None, 
    #                                                       arg.batch_size,
    #                                                       shuffle=split == 'train', 
    #                                                       num_workers=arg.num_workers,
    #                                                       drop_last=True,
    #                                                       max_num_samples=max_num_samples,
    #                                                       use_lmdb=arg.use_lmdb,
    #                                                       suff=arg.suff)
    
    # datasets[split].__getitem__(69)

    
    
    
