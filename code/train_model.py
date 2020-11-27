#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 17:28:29 2020

@author: jingtao
"""

import pickle
import sys
import timeit
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import math
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from torch.autograd import Variable




### model:
class CompoundProteinInteractionPrediction(nn.Module):
    def __init__(self):
        super(CompoundProteinInteractionPrediction, self).__init__()
        
        
        ### hyperparameters:        
        self.n_encoder=n_encoder
        self.n_decoder=n_decoder
        self.heads=heads
        self.dropout = dropout
        self.dim=dim
        self.dim_gnn=10
        self.d_ff=d_ff        
        
        ### models
        # gnn:
        self.embed_fingerprint = nn.Embedding(n_fingerprint, self.dim_gnn)
        self.W_gnn = nn.ModuleList([nn.Linear(self.dim_gnn, self.dim_gnn)
                                    for _ in range(layer_gnn)])
        
        # cnn:
        self.W_cnn = nn.ModuleList([nn.Conv2d(
                     in_channels=1, out_channels=1, kernel_size=2*11+1,
                     stride=1, padding=11) for _ in range(3)])
        ## multi channel
        '''
        self.W_cnn = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=2, 
                      kernel_size=2*window+1,
                     stride=1, padding=window),
            nn.Conv2d(in_channels=2, out_channels=2, 
                      kernel_size=2*window+1,
                     stride=1, padding=window),
            nn.Conv2d(in_channels=2, out_channels=1, 
                      kernel_size=2*window+1,
                     stride=1, padding=window),
            ])
        '''
        
        
        # transformer:
        self.embed_word = nn.Embedding(n_word, self.dim)
        self.positional_encoder=positional_encoder(self.dim,self.dropout)
        self.encoder=encoder(self.n_encoder, self.dim, self.d_ff, self.dropout, heads=self.heads)
        self.decoder=decoder(self.n_decoder, self.dim, self.d_ff, self.dropout, heads=self.heads)
        
        
        # interaction:
        self.W_out = nn.ModuleList([nn.Linear(self.dim_gnn+self.dim, self.dim_gnn+self.dim)
                                    for _ in range(layer_output)])
        self.W_interaction = nn.Linear(self.dim_gnn+self.dim, 2)
        
        

    def gnn(self, xs, A, layer):
        for i in range(layer):
            hs = torch.relu(self.W_gnn[i](xs))
            xs = xs + torch.matmul(A, hs)
        # return torch.unsqueeze(torch.sum(xs, 0), 0)
        return torch.unsqueeze(torch.mean(xs, 0), 0)
    
    def attention_cnn(self, x, xs, layer):
        """The attention mechanism is applied to the last layer of CNN."""

        xs = torch.unsqueeze(torch.unsqueeze(xs, 0), 0)
        for i in range(layer):
            xs = torch.relu(self.W_cnn[i](xs))
        xs = torch.squeeze(torch.squeeze(xs, 0), 0)

        return xs
        
    def transformer(self, compound, protein, n_encoder, n_decoder, heads):
        protein=self.positional_encoder(protein)
        protein=self.encoder(protein)
        protein = self.attention_cnn(protein,protein,3)
        protein=self.decoder(protein, compound)
        
        return protein
        
    def forward(self, inputs):

        fingerprints, adjacency, words = inputs

        """Compound vector with GNN."""
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        compound_vector = self.gnn(fingerprint_vectors, adjacency, layer_gnn)

        """Protein vector with transformer."""
        
        word_vectors = self.embed_word(words)
        
        protein_vector = self.transformer(compound_vector,
                        word_vectors,self.n_encoder,self.n_decoder,self.heads)

        """Concatenate the above two vectors and output the interaction."""
        cat_vector = torch.cat((compound_vector, protein_vector), 1)
        interaction = self.W_interaction(cat_vector)

        return interaction

    def __call__(self, data, train=True):
        inputs, correct_interaction = data[:-1], data[-1]
        predicted_interaction = self.forward(inputs)

        if train:
            loss = F.cross_entropy(predicted_interaction, correct_interaction)
            return loss
        else:
            correct_labels = correct_interaction.to('cpu').data.numpy()
            ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            return correct_labels, predicted_labels, predicted_scores

# end of the model


### transformer:
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class positional_encoder(nn.Module):
    def __init__(self, d_model, dropout, max_len=13100):
        super(positional_encoder, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2)).type(torch.FloatTensor) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position.type(torch.FloatTensor) * div_term.type(torch.FloatTensor))
        pe[:, 1::2] = torch.cos(position.type(torch.FloatTensor) * div_term.type(torch.FloatTensor))
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x):
        x = x + Variable(self.pe[0:x.size(0), :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)


class encoder(nn.Module):
    def __init__(self, n, dim, d_ff, dropout, heads):
        super(encoder, self).__init__()
        self.layers = clones(encoder_layer(dim, heads, 
                             self_attn(heads, dim, dropout).to(device), 
                             PositionwiseFeedForward(dim, d_ff), dropout), n)
        
    def forward(self, x, mask=None): 
        for layer in self.layers:
            x = layer(x, mask)
        return x

class decoder(nn.Module):
    def __init__(self, n, dim, d_ff, dropout, heads):
        super(decoder, self).__init__()
        self.layers = clones(decoder_layer(dim, heads, 
                             tgt_attn(heads, dim,dropout).to(device), 
                             self_attn(heads, dim, dropout).to(device),
                             PositionwiseFeedForward(dim, d_ff), dropout), n)
        self.tgt_out = tgt_out(heads, dim, dropout)
        self.final_norm = LayerNorm(dim)
    def forward(self, x, tgt):
        for layer in self.layers:
            x = layer(x, tgt)
        x=self.tgt_out(tgt,x,x)
        x=self.final_norm(x)
        return x

## attentions:
class self_attn(nn.Module):
    def __init__(self, h, dim, dropout=0):
        super(self_attn, self).__init__()
        assert dim % h == 0
        # We assume d_v always equals d_k
        self.d_k = dim // h
        self.h = h
        self.linears = clones(nn.Linear(dim, dim), 3)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None): 
        if mask is not None:
            mask = mask.unsqueeze(1)
        nwords = key.size(0) 
      
        query, key, value = \
            [l(x).view(nwords, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        # qkv.size() = length,heads,1,dk
            
        query=query.squeeze(2).transpose(0,1)               # heads, length, dk
        key=key.squeeze(2).transpose(0,1).transpose(1,2)    # heads, dk, length
        value=value.squeeze(2).transpose(0,1)               # heads, length, dk
        
        scores = torch.matmul(query,key)                    # heads, length, length
        p_attn = F.softmax(scores, dim = 2)                 # heads, length, length
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
        
        x=torch.matmul(p_attn, value)                       # heads, length, dk
        x=x.transpose(0,1).contiguous().view([nwords,self.h * self.d_k]) 
        #x=x.transpose(0,1).view([nwords,self.h * self.d_k]) 
        self.attn=p_attn  
        
        return self.linears[-1](x).unsqueeze(1)  

class tgt_out(nn.Module):    
    def __init__(self, h, dim, dropout=0):
        super(tgt_out, self).__init__()
        assert dim % h == 0
        # We assume d_v always equals d_k
        self.d_k = dim // h
        self.h = h
        self.tgt_linear = nn.Linear(10,dim)
        self.linears = clones(nn.Linear(dim, dim), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, query, key, value, mask=None):   #q=tgt, k=self, v=self 
        nwords = key.size(0) 
        query = self.tgt_linear(query) # from gnn_dim to dim
        query = self.linears[0](query).view(-1,self.h,self.d_k).transpose(0,1)        # heads, 1, dk
        key   = self.linears[1](key).view(nwords,-1,self.h,self.d_k).transpose(1,2)   # length, heads, 1, dk
        value = self.linears[2](value).view(nwords,-1,self.h,self.d_k).transpose(1,2) # length, heads, 1, dk

        key=key.squeeze(2).transpose(0,1).transpose(1,2)    # heads, dk, length
        scores = torch.matmul(query,key) 
        p_attn = F.softmax(scores, dim = 2)     # heads,1,length

        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
        value=value.squeeze(2).transpose(0,1)   # heads,length,dk 
        
        x=torch.matmul(p_attn, value)         
        x=x.transpose(0,1).contiguous().view([1,self.h * self.d_k]) 
        self.attn=p_attn  
        
        return self.linears[-1](x) 
    
class tgt_attn(nn.Module):    
    def __init__(self, h, dim, dropout=0):
        super(tgt_attn, self).__init__()
        assert dim % h == 0
        # We assume d_v always equals d_k
        self.d_k = dim // h
        self.h = h
        self.tgt_linear = nn.Linear(10,dim)
        self.linears = clones(nn.Linear(dim, dim), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, query, key, value, mask=None):   #q=tgt, k=self, v=self 
        nwords = key.size(0) 
        query = self.tgt_linear(query) # from gnn_dim to dim
        query = self.linears[0](query).view(-1,self.h,self.d_k).transpose(0,1)        # heads, 1, dk
        key   = self.linears[1](key).view(nwords,-1,self.h,self.d_k).transpose(1,2)   # length, heads, 1, dk
        value = self.linears[2](value).view(nwords,-1,self.h,self.d_k).transpose(1,2) # length, heads, 1, dk

        key=key.squeeze(2).transpose(0,1).transpose(1,2)    # heads, dk, length
        scores = torch.matmul(query,key) 
        p_attn = F.softmax(scores, dim = 2).transpose(1,2)     # heads,length,1

        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
        
        value=value.squeeze(2).transpose(0,1)                  # heads,length,dk
        
        x=p_attn*value       #  heads,length,dk
        x=x.transpose(0,1).contiguous().view([nwords,self.h * self.d_k]) 
        self.attn=p_attn     #  length, dim
        
        return self.linears[-1](x) 
## end of attentions


## encoder & decoder layers
class encoder_layer(nn.Module):
    def __init__(self, dim, heads, self_attn, feedforward, dropout):  
        super(encoder_layer, self).__init__()
        self.res_layer = [residual_layer( dim, dropout,self_attn),
                          residual_layer( dim, dropout,feedforward)]
        self.dim=dim
    def forward(self, x, mask=None):
        x = self.res_layer[0](x,x,x)
        return self.res_layer[1](x)
    
    
class decoder_layer(nn.Module):
    def __init__(self, dim, heads, tgt_attn, self_attn, feedforward, dropout):  
        super(decoder_layer, self).__init__()
        self.res_layer = [residual_layer( dim, dropout,tgt_attn),
                          residual_layer( dim, dropout,self_attn),
                          residual_layer( dim, dropout,feedforward)]

    def forward(self, x,tgt):
        x = self.res_layer[0](x,tgt,x)  # res_layer: v, q, k
        x = self.res_layer[1](x,x,x)        
        return self.res_layer[2](x)
## end of encoder & decoder layers

class residual_layer(nn.Module):
    def __init__(self, size, dropout, sublayer):
        super(residual_layer, self).__init__()
        self.norm = LayerNorm(size).to(device)
        self.dropout = nn.Dropout(dropout)
        self.sublayer=sublayer
        
    def forward(self,x, q=None, k=None ):  # q and k are None if sublayer is ff, x is v
        if (q!=None and k!=None):
            return self.norm(x+self.dropout(self.sublayer(q,k,x).squeeze(1)))
        else:
            return self.norm(x+self.dropout(self.sublayer(x).squeeze(1)))
    
    
class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        #return norm+bias
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2   
    
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff).to(device)
        self.w_2 = nn.Linear(d_ff, d_model).to(device)
        self.dropout = nn.Dropout(dropout).to(device)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
    
### end of transformer


class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=1e-7, weight_decay=weight_decay) ##start with warm up rate

    def train(self, dataset):
        np.random.shuffle(dataset)
        N = len(dataset)
        loss_total = 0
        for data in dataset:
            loss = self.model(data)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.to('cpu').data.numpy()
        return loss_total


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataset):
        N = len(dataset)
        T, Y, S = [], [], []
        for data in dataset:
            (correct_labels, predicted_labels,
             predicted_scores) = self.model(data, train=False)
            T.append(correct_labels)
            Y.append(predicted_labels)
            S.append(predicted_scores)
        AUC = roc_auc_score(T, S)
        precision = precision_score(T, Y)
        recall = recall_score(T, Y)
        return AUC, precision, recall

    def save_AUCs(self, AUCs, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, AUCs)) + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)

def rm_long(dataset,length):
    d=[]
    for i in dataset:
        if (i[2].size()[0]<length):
            d.append(i)
    return d

def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy')]


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2


if __name__ == "__main__":

    """Hyperparameters."""
    (DATASET,dataname, radius, ngram, dim, d_ff, layer_gnn, layer_output,heads, n_encoder,n_decoder,
     lr, lr_decay, decay_interval, weight_decay, iteration, warmup_step,
     dropout,setting) = sys.argv[1:]

    (dim, d_ff, layer_gnn, layer_output, decay_interval,
     iteration,heads, n_encoder,n_decoder,warmup_step) = map(int, [dim, d_ff, layer_gnn, layer_output,
                            decay_interval, iteration,heads, n_encoder,n_decoder,warmup_step])
                            
              
    lr, lr_decay, weight_decay,dropout = map(float, [lr, lr_decay, weight_decay,dropout])

    """CPU or GPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    """Load preprocessed data."""
    dir_input = ('../dataset/' + DATASET + '/input/'
                 'radius' + radius + '_ngram' + ngram + ' '+dataname+'/')
    compounds = load_tensor(dir_input + 'compounds', torch.LongTensor)
    adjacencies = load_tensor(dir_input + 'adjacencies', torch.FloatTensor)
    proteins = load_tensor(dir_input + 'proteins', torch.LongTensor)
    interactions = load_tensor(dir_input + 'interactions', torch.LongTensor)
    fingerprint_dict = load_pickle(dir_input + 'fingerprint_dict.pickle')
    word_dict = load_pickle(dir_input + 'word_dict.pickle')
    n_fingerprint = len(fingerprint_dict)
    n_word = len(word_dict)

    """Create a dataset and split it into train/dev/test."""
    dataset = list(zip(compounds, adjacencies, proteins, interactions))
    dataset = rm_long(dataset,6000)
    dataset = shuffle_dataset(dataset, 1234)
    dataset_train, dataset_ = split_dataset(dataset, 0.8)
    dataset_dev, dataset_test = split_dataset(dataset_, 0.5)

    """Set a model."""
    torch.manual_seed(1234)
    model = CompoundProteinInteractionPrediction().to(device)
    trainer = Trainer(model)
    tester = Tester(model)

    """Output files."""
    file_AUCs = '../output/result/AUCs--' +dataname +" "+setting + '.txt'
    file_model = '../output/model/' +dataname+" " +setting
    AUCs = ('Epoch\tTime(sec)\tLoss_train\tAUC_dev\t'
            'AUC_test\tPrecision_test\tRecall_test')
    with open(file_AUCs, 'w') as f:
        f.write(AUCs + '\n')

    """Start training."""
    print('Training...')
    print(AUCs)
    start = timeit.default_timer()


 
    for epoch in range(1, warmup_step):
        
        trainer.optimizer.param_groups[0]['lr'] += (lr-1e-7)/warmup_step
        loss_train = trainer.train(dataset_train)
        AUC_dev = tester.test(dataset_dev)[0]
        AUC_test, precision_test, recall_test = tester.test(dataset_test)
        end = timeit.default_timer()
        time = end - start
        AUCs = [epoch, time, loss_train, AUC_dev,
                AUC_test, precision_test, recall_test]
        tester.save_AUCs(AUCs, file_AUCs)
        tester.save_model(model, file_model)
        print('\t'.join(map(str, AUCs)))
        
    for epoch in range(1, iteration):

        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay

        loss_train = trainer.train(dataset_train)
        AUC_dev = tester.test(dataset_dev)[0]
        AUC_test, precision_test, recall_test = tester.test(dataset_test)

        end = timeit.default_timer()
        time = end - start

        AUCs = [epoch, time, loss_train, AUC_dev,
                AUC_test, precision_test, recall_test]
        tester.save_AUCs(AUCs, file_AUCs)
        tester.save_model(model, file_model)

        print('\t'.join(map(str, AUCs)))
