#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 17:28:29 2020

@author: jingtao
"""


### model 1 self attention -> attn_cnn

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
#from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from math import sqrt
from scipy import stats
from sklearn import preprocessing,metrics




### metrics
def rmse(y,f):
    """
    Task:    To compute root mean squared error (RMSE)
    Input:   y      Vector with original labels (pKd [M])
             f      Vector with predicted labels (pKd [M])
    Output:  rmse   RSME
    """
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse

def pearson(y,f):
    """
    Task:    To compute Pearson correlation coefficient
    Input:   y      Vector with original labels (pKd [M])
             f      Vector with predicted labels (pKd [M])
    Output:  rp     Pearson correlation coefficient
    """
    rp = np.corrcoef(y, f)[0,1]
    return rp

def spearman(y,f):
    """
    Task:    To compute Spearman's rank correlation coefficient
    Input:   y      Vector with original labels (pKd [M])
             f      Vector with predicted labels (pKd [M])
    Output:  rs     Spearman's rank correlation coefficient
    """
    rs = stats.spearmanr(y, f)[0]
    return rs

def reg_scores(label, pred):
	label = label.reshape(-1)
	pred = pred.reshape(-1)
	return rmse(label, pred), pearson(label, pred), spearman(label, pred)
### end of metrics






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
        self.W_attention = nn.Linear(self.dim, self.dim)
        
        
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
        self.W_interaction = nn.Linear(self.dim_gnn+self.dim, 1)
        
        self.mse=nn.MSELoss()
        

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

        h = torch.relu(self.W_attention(x))
        hs = torch.relu(self.W_attention(xs))
        weights = torch.tanh(F.linear(h, hs))
        ys = torch.t(weights) * hs

        # return torch.unsqueeze(torch.sum(ys, 0), 0)
        return torch.unsqueeze(torch.mean(ys, 0), 0)
        #return xs
    def transformer(self, compound, protein, n_encoder, n_decoder, heads):
        protein=self.positional_encoder(protein)
        
        if (n_encoder!=0):
            protein=self.encoder(protein)
        
        #if (n_decoder!=0):
        #    protein=self.decoder(protein, compound)
        
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
        
        # attention CNN
        protein_vector = self.attention_cnn(compound_vector,protein_vector,3)
        
        """Concatenate the above two vectors and output the interaction."""
        cat_vector = torch.cat((compound_vector, protein_vector), 1)
        #for j in range(layer_output):
        #    cat_vector = torch.relu(self.W_out[j](cat_vector))
        interaction = self.W_interaction(cat_vector)

        return interaction

    def __call__(self, data, train=True):
        inputs, correct_interaction = data[:-1], data[-1]
        predicted_interaction = self.forward(inputs)#.double()

        if train:
            correct_interaction=correct_interaction.reshape(1,1)
            #print('5',type(predicted_interaction.item()))
            #print('4',type(correct_interaction.item()))
            loss = self.mse(predicted_interaction, correct_interaction.to(device))
            loss=loss#.double()
            #print('3',type(loss.item()))
            #loss = F.cross_entropy(predicted_interaction, correct_interaction)
            return loss
        else:
           # correct_labels = correct_interaction.to('cpu').data.numpy()
           # ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
           # predicted_labels = list(map(lambda x: np.argmax(x), ys))
           # predicted_scores = list(map(lambda x: x[1], ys))
            #rmse_value, pearson_value, spearman_value = reg_scores(correct_interaction, predicted_interaction)
            return correct_interaction, predicted_interaction

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

    def train(self, datase):
        np.random.shuffle(dataset)
        N = len(dataset)
        loss_total = 0
        for data in dataset:
            loss = self.model(data)
            #print('2',type(loss.item()))
            #loss.float()
            self.optimizer.zero_grad()
            #loss.float()
            #print('1',type(loss.item()))
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
            (correct_labels, predicted_labels) = self.model(data, train=False)
            T.append(correct_labels)
            Y.append(predicted_labels)
            #S.append(spearman_value)
        AUC = roc_auc_score(T, S)
        #precision = precision_score(T, Y)
        #recall = recall_score(T, Y)
        rmse_value, pearson_value, spearman_value = reg_scores(T, Y)
        return rmse_value, pearson_value, spearman_value,AUC

    def save_AUCs(self, AUCs, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, AUCs)) + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)







def split_train_test_clusters(measure, clu_thre, n_fold):
	# load cluster dict
	cluster_path = '../dataset/preprocessing/'
	with open(cluster_path+measure+'_compound_cluster_dict_'+str(clu_thre), 'rb') as f:
		C_cluster_dict = pickle.load(f)
	with open(cluster_path+measure+'_protein_cluster_dict_'+str(clu_thre), 'rb') as f:
		P_cluster_dict = pickle.load(f)
	
	C_cluster_set = set(list(C_cluster_dict.values()))
	P_cluster_set = set(list(P_cluster_dict.values()))
	C_cluster_list = np.array(list(C_cluster_set))
	P_cluster_list = np.array(list(P_cluster_set))
	np.random.shuffle(C_cluster_list)
	np.random.shuffle(P_cluster_list)
 
    
	# n-fold split
	#c_kf = KFold(len(C_cluster_list), n_fold, shuffle=True)
	#p_kf = KFold(len(P_cluster_list), n_fold, shuffle=True)
	c_kf = KFold(n_fold,shuffle=True)
    
	p_kf = KFold(n_fold,shuffle=True)
	c_train_clusters, c_test_clusters = [], []
	for train_idx, test_idx in c_kf.split(C_cluster_list):
		c_train_clusters.append(C_cluster_list[train_idx])
		c_test_clusters.append(C_cluster_list[test_idx])
	p_train_clusters, p_test_clusters = [], []
	for train_idx, test_idx in p_kf.split(P_cluster_list):
		p_train_clusters.append(P_cluster_list[train_idx])
		p_test_clusters.append(P_cluster_list[test_idx])
	
	
	pair_kf = KFold(n_fold,shuffle=True)
	pair_list = []
	for i_c in C_cluster_list:
		for i_p in P_cluster_list:
			pair_list.append('c'+str(i_c)+'p'+str(i_p))
	pair_list = np.array(pair_list)
	np.random.shuffle(pair_list)
	#pair_kf = KFold(len(pair_list), n_fold, shuffle=True)
	pair_train_clusters, pair_test_clusters = [], []
	for train_idx, test_idx in pair_kf.split(pair_list):
		pair_train_clusters.append(pair_list[train_idx])
		pair_test_clusters.append(pair_list[test_idx])
	
	return pair_train_clusters, pair_test_clusters, c_train_clusters,\
        c_test_clusters, p_train_clusters, p_test_clusters, C_cluster_dict, P_cluster_dict


def load_data(measure, setting, clu_thre, n_fold):
	# load data
	with open('../dataset/preprocessing/pdbbind_all_combined_input_'+measure,'rb') as f:
		data_pack = pickle.load(f)
	cid_list = data_pack[9]
	pid_list = data_pack[10]
	n_sample = len(cid_list)
	
	# train-test split
	train_idx_list, valid_idx_list, test_idx_list = [], [], []
	print('setting: ',setting)
	if setting == 'imputation':
		pair_train_clusters, pair_test_clusters, c_train_clusters, c_test_clusters,\
            p_train_clusters, p_test_clusters, C_cluster_dict, P_cluster_dict \
		= split_train_test_clusters(measure, clu_thre, n_fold)
		for fold in range(n_fold):
			pair_train_valid, pair_test = pair_train_clusters[fold], pair_test_clusters[fold]
			pair_valid = np.random.choice(pair_train_valid, int(len(pair_train_valid)*0.125), replace=False)
			pair_train = set(pair_train_valid)-set(pair_valid)
			pair_valid = set(pair_valid)
			pair_test = set(pair_test)
			train_idx, valid_idx, test_idx = [], [], []
			for ele in range(n_sample):
				if 'c'+str(C_cluster_dict[cid_list[ele]])+'p'+str(P_cluster_dict[pid_list[ele]]) in pair_train:
					train_idx.append(ele)
				elif 'c'+str(C_cluster_dict[cid_list[ele]])+'p'+str(P_cluster_dict[pid_list[ele]]) in pair_valid:
					valid_idx.append(ele)
				elif 'c'+str(C_cluster_dict[cid_list[ele]])+'p'+str(P_cluster_dict[pid_list[ele]]) in pair_test:
					test_idx.append(ele)
				else:
					print('error')
			train_idx_list.append(train_idx)
			valid_idx_list.append(valid_idx)
			test_idx_list.append(test_idx)
			print('fold '+ fold+ 'train '+len(train_idx)+'test ',len(test_idx),'valid ',len(valid_idx))
			
	elif setting == 'new_protein':
		pair_train_clusters, pair_test_clusters, c_train_clusters, c_test_clusters, p_train_clusters, p_test_clusters, C_cluster_dict, P_cluster_dict \
		= split_train_test_clusters(measure, clu_thre, n_fold)
		for fold in range(n_fold):
			p_train_valid, p_test = p_train_clusters[fold], p_test_clusters[fold]
			p_valid = np.random.choice(p_train_valid, int(len(p_train_valid)*0.125), replace=False)
			p_train = set(p_train_valid)-set(p_valid)
			train_idx, valid_idx, test_idx = [], [], []
			for ele in range(n_sample): 
				if P_cluster_dict[pid_list[ele]] in p_train:
					train_idx.append(ele)
				elif P_cluster_dict[pid_list[ele]] in p_valid:
					valid_idx.append(ele)
				elif P_cluster_dict[pid_list[ele]] in p_test:
					test_idx.append(ele)
				else:
					print('error')
			train_idx_list.append(train_idx)
			valid_idx_list.append(valid_idx)
			test_idx_list.append(test_idx)
			print ('fold', fold, 'train ',len(train_idx),'test ',len(test_idx),'valid ',len(valid_idx))
			
	elif setting == 'new_compound':
		pair_train_clusters, pair_test_clusters, c_train_clusters, c_test_clusters, p_train_clusters, p_test_clusters, C_cluster_dict, P_cluster_dict \
		= split_train_test_clusters(measure, clu_thre, n_fold)
		for fold in range(n_fold):
			c_train_valid, c_test = c_train_clusters[fold], c_test_clusters[fold]
			c_valid = np.random.choice(c_train_valid, int(len(c_train_valid)*0.125), replace=False)
			c_train = set(c_train_valid)-set(c_valid)
			train_idx, valid_idx, test_idx = [], [], []
			for ele in range(n_sample):
				if C_cluster_dict[cid_list[ele]] in c_train:
					train_idx.append(ele)
				elif C_cluster_dict[cid_list[ele]] in c_valid:
					valid_idx.append(ele)
				elif C_cluster_dict[cid_list[ele]] in c_test:
					test_idx.append(ele)
				else:
					print('error')
			train_idx_list.append(train_idx)
			valid_idx_list.append(valid_idx)
			test_idx_list.append(test_idx)
			print ('fold', fold, 'train ',len(train_idx),'test ',len(test_idx),'valid ',len(valid_idx))
	
	elif setting == 'new_new':
		assert n_fold ** 0.5 == int(n_fold ** 0.5)
		pair_train_clusters, pair_test_clusters, c_train_clusters, c_test_clusters, p_train_clusters, p_test_clusters, C_cluster_dict, P_cluster_dict \
		= split_train_test_clusters(measure, clu_thre, int(n_fold ** 0.5))
		
		for fold_x in range(int(n_fold ** 0.5)):
			for fold_y in range(int(n_fold ** 0.5)):
				c_train_valid, p_train_valid = c_train_clusters[fold_x], p_train_clusters[fold_y]
				c_test, p_test = c_test_clusters[fold_x], p_test_clusters[fold_y]
				c_valid = np.random.choice(list(c_train_valid), int(len(c_train_valid)/3), replace=False)
				c_train = set(c_train_valid)-set(c_valid)
				p_valid = np.random.choice(list(p_train_valid), int(len(p_train_valid)/3), replace=False)
				p_train = set(p_train_valid)-set(p_valid)
				
				train_idx, valid_idx, test_idx = [], [], []
				for ele in range(n_sample):
					if C_cluster_dict[cid_list[ele]] in c_train and P_cluster_dict[pid_list[ele]] in p_train:
						train_idx.append(ele)
					elif C_cluster_dict[cid_list[ele]] in c_valid and P_cluster_dict[pid_list[ele]] in p_valid:
						valid_idx.append(ele)
					elif C_cluster_dict[cid_list[ele]] in c_test and P_cluster_dict[pid_list[ele]] in p_test:
						test_idx.append(ele)
				train_idx_list.append(train_idx)
				valid_idx_list.append(valid_idx)
				test_idx_list.append(test_idx)
				print ('fold', fold_x*int(n_fold ** 0.5)+fold_y, 'train ',
           len(train_idx),'test ',len(test_idx),'valid ',len(valid_idx))
	return data_pack, train_idx_list, valid_idx_list, test_idx_list



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

def datalist(data,l):
    d=[]
    for i in l:
        #d.append([data[0][i],data[1][i],data[2][i],data[3][i]])
        d.append(data[i])
    return d

if __name__ == "__main__":

    """Hyperparameters."""
    (DATASET,dataname, radius, ngram, dim, d_ff, layer_gnn, layer_output,heads, n_encoder,n_decoder,
     lr, lr_decay, decay_interval, weight_decay, iteration, warmup_step,
     dropout,setting) = ('monn','_monn','2','3','10','10','3','1','2','2','1','1e-4','0.5','10',
                         '1e-6','100','20','0.1','qwe')
    #sys.argv[1:]
    
    
    
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
                 'radius' + radius + '_ngram' + ngram +dataname+'/')
    compounds = load_tensor(dir_input + 'compounds', torch.LongTensor)
    adjacencies = load_tensor(dir_input + 'adjacencies', torch.FloatTensor)
    proteins = load_tensor(dir_input + 'proteins', torch.LongTensor)
    #interactions = load_tensor(dir_input + 'interactions', torch.FloatTensor)  ## longtensor for tsubaki, float for monn
    interactions = torch.from_numpy(np.load(dir_input + 'interactions.npy'))
    fingerprint_dict = load_pickle(dir_input + 'fingerprint_dict.pickle')
    word_dict = load_pickle(dir_input + 'word_dict.pickle')
    n_fingerprint = len(fingerprint_dict)
    n_word = len(word_dict)

    """Create a dataset and split it into train/dev/test."""
    #dataset = list(zip(compounds, adjacencies, proteins, interactions))
    
    """Set a model."""
    torch.manual_seed(1234)
    model = CompoundProteinInteractionPrediction().to(device)
    trainer = Trainer(model)
    tester = Tester(model)
    

 	#evaluate scheme
    measure = 'KIKD'  # IC50 or KIKD
    setting = 'new_compound'  # new_compound, new_protein or new_new
    clu_thre = 0.3  # 0.3, 0.4, 0.5 or 0.6
    n_epoch = 30
    n_rep = 1#10
	
    assert setting in ['new_compound', 'new_protein', 'new_new']
    assert clu_thre in [0.3, 0.4, 0.5, 0.6]
    assert measure in ['IC50', 'KIKD']
    GNN_depth, inner_CNN_depth, DMA_depth = 4, 2, 2
    if setting == 'new_compound':
        n_fold = 5
        batch_size = 32
        k_head, kernel_size, hidden_size1, hidden_size2 = 2, 7, 128, 128
    elif setting == 'new_protein':
        n_fold = 5
        batch_size = 32
        ead, kernel_size, hidden_size1, hidden_size2 = 1, 5, 128, 128
    elif setting == 'new_new':
        n_fold = 9
        batch_size = 32
        #k_head, kernel_size, hidden_size1, hidden_size2 = 1, 7, 128, 128
    para_names = ['GNN_depth', 'inner_CNN_depth', 'DMA_depth', 'k_head', 'kernel_size', 'hidden_size1', 'hidden_size2']
	
	#params = [GNN_depth, inner_CNN_depth, DMA_depth, k_head, kernel_size, hidden_size1, hidden_size2]
	#params = sys.argv[4].split(',')
	#params = map(int, params)
	
	#print evaluation scheme
    print('Dataset: PDBbind v2018 with measurement', measure)
    print('Clustering threshold:', clu_thre)
    print('Number of epochs:', n_epoch)
    print('Number of repeats:', n_rep)
    #print('Hyper-parameters:', [para_names[i]+':'+str(params[i]) for i in range(7)])
    file_AUCs = '../output/result/AUCs--' +dataname +" "+setting + '.txt'
    file_model = '../output/model/' +dataname+" " +setting
    AUCs = ('Epoch\tTime(sec)\tLoss_train\tAUC_dev\t'
            'AUC_test\tPrecision_test\tRecall_test')
    with open(file_AUCs, 'w') as f:
        f.write(AUCs + '\n')

    """Start training."""
    print('Training...')
    print(AUCs)
    rep_all_list = []
    rep_avg_list = []
    for a_rep in range(n_rep):
        #load data
        data_pack, train_idx_list, valid_idx_list, test_idx_list = load_data(measure, setting, clu_thre, n_fold)
        #dataset = list(zip(compounds, adjacencies, proteins, interactions))
        dataset=[data_pack[0],data_pack[1],data_pack[7],data_pack[8]]  #dtype(d).to(device)
        dataset=list(np.array(dataset).T)
        for i in range(len(dataset)):
            dataset[i][0]=torch.LongTensor(dataset[i][0]).to(device)
            dataset[i][1]=torch.FloatTensor(dataset[i][1]).to(device)
            dataset[i][2]=torch.LongTensor(dataset[i][2]).to(device)
            dataset[i][3]=torch.FloatTensor([dataset[i][3]]).to(device)
        
        d_score_list = []
        fold_score_list = []
        for a_fold in range(n_fold):
            print ('repeat', a_rep+1, 'fold', a_fold+1, 'begin')
            train_idx, valid_idx, test_idx = train_idx_list[a_fold], valid_idx_list[a_fold], \
                test_idx_list[a_fold]
            print ('train num:', len(train_idx), 'valid num:', len(valid_idx), 'test num:', len(test_idx))
			
            train_data = datalist(dataset,train_idx)
            valid_data = datalist(dataset,valid_idx)
            test_data = datalist(dataset,test_idx)
            #train_data = dataset[train_idx]
            #valid_data = dataset[valid_idx]
            #test_data = dataset[test_idx]
			
            start = timeit.default_timer()
            for epoch in range(1, warmup_step):
        
                trainer.optimizer.param_groups[0]['lr'] += (lr-1e-7)/warmup_step
                loss_train = (trainer.train(train_data))
                rmse_value_dev, pearson_value_dev, spearman_value_dev,AUC_dev = tester.test(valid_data)
                rmse_value, pearson_value, spearman_value,AUC = tester.test(test_data)
                end = timeit.default_timer()
                time = end - start
                AUCs = [epoch, time, loss_train, AUC_dev,
                        rmse_value_dev, pearson_value_dev, spearman_value_dev,AUC_dev,
                        rmse_value, pearson_value, spearman_value,AUC]
                tester.save_AUCs(AUCs, file_AUCs)
                tester.save_model(model, file_model)
                print(' '.join(map(str, AUCs)))
        
            for epoch in range(1, iteration):

                if epoch % decay_interval == 0:
                        trainer.optimizer.param_groups[0]['lr'] *= lr_decay

                loss_train = trainer.train(train_data)
                rmse_value_dev, pearson_value_dev, spearman_value_dev,AUC_dev = tester.test(valid_data)
                rmse_value, pearson_value, spearman_value,AUC = tester.test(test_data)
                end = timeit.default_timer()
                time = end - start
                AUCs = [epoch, time, loss_train, AUC_dev,
                        rmse_value_dev, pearson_value_dev, spearman_value_dev,AUC_dev,
                        rmse_value, pearson_value, spearman_value,AUC]
                tester.save_AUCs(AUCs, file_AUCs)
                tester.save_model(model, file_model)

                print(' '.join(map(str, AUCs)))
                
            test_performance = [rmse_value, pearson_value, spearman_value, AUC]
            
    		#test_performance, test_label, test_output 
            
            rep_all_list.append(test_performance)
            fold_score_list.append(test_performance)
            print ('-'*30)
        print ('fold avg performance', np.mean(fold_score_list,axis=0))
        rep_avg_list.append(np.mean(fold_score_list,axis=0))
        np.save('MONN_rep_all_list_'+measure+'_'+setting+'_thre'+str(clu_thre), rep_all_list)
    
    
    
    
    
    '''
    for epoch in range(1, warmup_step):
        
        trainer.optimizer.param_groups[0]['lr'] += (lr-1e-7)/warmup_step
        loss_train = (trainer.train(dataset_train))
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
'''