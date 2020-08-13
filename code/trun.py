import pickle
import sys
import timeit

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import math
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from torch.autograd import Variable


#parameters
'''
radius=2

ngram=3

dim=8  

d_ff # position-wise feed forward layer dimension

N  # the number of self-attn layers 

M  # the number of tgt_attn layers 

layer_gnn=3

window=5  # The window size is 2*window+1.

layer_cnn=3

lr=1e-4

lr_decay=0.5

decay_interval=10

iteration=100
'''


class CompoundProteinInteractionPrediction(nn.Module):
    def __init__(self):
        super(CompoundProteinInteractionPrediction, self).__init__()
        self.embed_fingerprint = nn.Embedding(n_fingerprint, 10)             # flaw1: should decode and get true structure? -JW
        self.embed_word = nn.Embedding(n_word, dim)
        self.W_gnn = nn.ModuleList([nn.Linear(10, 10)
                                    for _ in range(layer_gnn)])
        self.W_cnn = nn.ModuleList([nn.Conv2d(
                     in_channels=1, out_channels=1, kernel_size=2*window+1,   # flaw2: should use conv1d ??
                     stride=1, padding=window) for _ in range(layer_cnn)])
        self.W_attention = nn.Linear(dim, dim)
        self.W_out = nn.Linear(10+dim, 2)
        self.match = nn.Linear(10, dim)
	#transformer hyperparameters:
        tgt_vocab = 2
        d_model=dim   # 512  in transformer --JW
        d_ff=dim_ff   # 2048 in transformer --JW
        d_ff2=dim_ff2
        h=heads       # 8    in transformer --JW
        dropout=0   # 0.1  in transformer --JW
	#generating transformer components
        c=copy.deepcopy
        ff=PositionwiseFeedForward(d_model, d_ff, dropout)
        ff2=PositionwiseFeedForward(d_model, d_ff2, dropout)
        self.ff3=PositionwiseFeedForward(d_model, d_ff, dropout)
        attn=MultiHeadedAttention(h, d_model)
        tgt_attn=tgt_MultiHeadedAttention(h, d_model)
        #attn=nn.DataParallel(attn)
        self.norm=LayerNorm(d_model)
        self.position=PositionalEncoding(d_model,dropout)     
        self.encoder=Encoder(self_EncoderLayer(d_model, c(attn), c(ff), dropout),
        tgt_OutputLayer(d_model,c(tgt_attn), c(ff2), dropout), N,M) 
        self.generator=Generator(d_model, tgt_vocab)

    def gnn(self, xs, A, layer):   # embedded fingerprints, adjacency, #layer  --JW                                            
        for i in range(layer):
            hs = torch.relu(self.W_gnn[i](xs))  # FC Layer on embeddings doesn't make much sense. --JW
            xs = xs + torch.matmul(A, hs)
        return torch.unsqueeze(torch.sum(xs, 0), 0)

    def cnn(self, xs, i):
        xs = torch.unsqueeze(torch.unsqueeze(xs, 0), 0)
        hs = torch.relu(self.W_cnn[i](xs))
        return torch.squeeze(torch.squeeze(hs, 0), 0)
    
    #def scnn(self, x_protein, i):

    def transformer(self, x_compound, x_words):
        x_words=self.position(x_words)  # positional encoding  --JW
        x_compound=self.match(x_compound) # dim chg from 10 to $dim
        x_protein=self.encoder(x_words, x_compound)
        #x_protein=self.ff3(x_protein)
        #maybe add xavier_uniform initialization later
        return x_protein

    def attention_cnn(self, x, xs, layer):  # x: compound  xs: protein
        for i in range(layer):
            #print("x;tgt:")
            #print(xs)
            #print(xs.shape)
            #print(x)
            #print(x.shape)
            #a=input()
            hs = self.cnn(xs, i)  # should be 1D Conv Layer rather than 2D 
            x = torch.relu(self.W_attention(x))
            hs = torch.relu(self.W_attention(hs))
            weights = torch.tanh(F.linear(x, hs))
            xs = torch.t(weights) * hs
        return torch.unsqueeze(torch.sum(xs, 0), 0)
        
    def forward(self, inputs):

        fingerprints, adjacency, words = inputs

        """Compound vector with GNN."""
        x_fingerprints = self.embed_fingerprint(fingerprints)
        x_compound = self.gnn(x_fingerprints, adjacency, layer_gnn)
     #   x_compound = self.norm(x_compound)
        """Protein vector with attention-CNN."""
        x_words = self.embed_word(words)
        #x_protein = self.attention_cnn(x_compound, x_words, layer_cnn)
        x_protein = self.transformer(x_compound, x_words)
        
        y_cat = torch.cat((x_compound, x_protein), 1)
        z_interaction = self.W_out(y_cat)

        return z_interaction

    def __call__(self, data, train=True):

        inputs, t_interaction = data[:-1], data[-1]
        z_interaction = self.forward(inputs)

        if train:
            loss = F.cross_entropy(z_interaction, t_interaction)
            return loss
        else:
            z = F.softmax(z_interaction, 1).to('cpu').data[0].numpy()
            t = int(t_interaction.to('cpu').data[0].numpy())
            return z, t

   
#transformer components:

class self_EncoderLayer(nn.Module):
    "made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(self_EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask=None):
        #print(x.shape)  [86,8]
        #a=input()
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        #print(x.shape)  [86,86,8]
        return self.sublayer[1](x, self.feed_forward)

class tgt_OutputLayer(nn.Module):
    "output protein feature vector"
    def __init__(self, size, tgt_attn, feed_forward, dropout):
        super(tgt_OutputLayer, self).__init__()
        self.tgt_attn = tgt_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 1)
        self.size = size
        self.norm=LayerNorm(size)
    def forward(self, x,tgt, mask=None):
        #x = self.norm(x+self.tgt_attn(tgt, x, x, mask))
        x = self.tgt_attn(tgt, x, x, mask)
        #x = self.sublayer[0](x, lambda x: self.tgt_attn(x, x, x, mask))
        return self.sublayer[0](x, self.feed_forward)
        #return x

class tgt_EncoderLayer(nn.Module):
    def __init__(self, size,self_attn, tgt_attn, feed_forward, dropout):
        super(tgt_EncoderLayer, self).__init__()
        self.tgt_attn = tgt_attn
        self.self_attn=self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x,tgt, mask=None):
        x = self.sublayer[0](x, lambda x: self.tgt_attn(tgt, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        #print(x.shape)                                                [86,8]
        #print(self.dropout(sublayer(self.norm(x))).shape)             [86,1,8]
        #print((x + self.dropout(sublayer(self.norm(x)))).shape)       [86,86,8]
        #a=input()    
        #return x + self.dropout(sublayer(self.norm(x))).squeeze(1)
        return self.norm(x+self.dropout(sublayer(x).squeeze(1))) #ERR prob need squeeze
    
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    def __init__(self, layer1, layer2, N,M):
        super(Encoder, self).__init__()
        self.layers1 = clones(layer1, N)
        self.layers2 = clones(layer2, M)
        self.norm = LayerNorm(layer1.size)
        
    def forward(self, x, tgt, mask=None): ##originally mask may != None  x=x_words, tgt=x_compound
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers1:
            x = layer(x, mask)
        for layer in self.layers2:
            #print("x;tgt:")
            #print(x)
            #print(x.shape)
            #print(tgt)
            #print(tgt.shape)
            x = layer(x, tgt, mask)
        #x=self.norm(x)
        #x=(x.sum(0)/(x.shape[0])).unsqueeze(0)  #[86,8] -> [1,8]
       
        return x

'''def attention(query, key, value, mask=None, dropout=0):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = 1)    # watch the dimension!!!  !!! !!! 1!! 1!! !!!
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn'''

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 3)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
        
    def forward(self, query, key, value, mask=None): 
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nwords = key.size(0) #nwords==86
        #print("QKV")
        #print(query)
        #print(query.shape)   #[86,8] for both    (8 is d_model)
        #print(key.shape)     #[86,8] for self_attn; [1,8] for tgt
        #print(value.shape)   #[86,8] for self_attn; [1,8] for tgt
        #print((query,key,value))
        #a=input()
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        '''if (query.size()[0]==1):  # means k&v are tgt's. now stack them from (1,dim) to (n,dim)
          lk=[]
          for i in range(nwords):
            lk.append(query)
          query=torch.stack(lk,0).squeeze(1)'''
      
        query, key, value = \
            [l(x).view(nwords, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        #print(query.shape) [86,2,1,4]
        #print(key.shape)   [86,2,1,4]
        #a=input()
        # 2) Apply attention on all the projected vectors in batch. 
        #x, self.attn = attention(query, key, value, mask=mask, 
        #                         dropout=self.dropout)
        d_k=query.size(-1)
        h=query.size(1)
        query=query.transpose(0,1) #[2,86,1,4]
        key=key.transpose(0,1)  # [2,86,1,4]
        query=query.squeeze(2)  #[2,86,4]
        key=key.squeeze(2).transpose(1,2)  # [2,4,86]
        #print(query.shape)
        #print(key.shape)
        scores = torch.matmul(query,key) #[2,86,86] 
        p_attn = F.softmax(scores, dim = 2) #[2,86,86]
        #print(scores.shape) [h,806,806]
        #print(p_attn.shape) {h,806,806}
        #a=input()
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
        value=value.squeeze(2).transpose(0,1)  #[2,86,4]
         
        
        
        x=torch.matmul(p_attn, value)  #[2,86,4]     CHECK TWICE!!! ATTN X VALUE  IMPORTANT!!!        
        x=x.transpose(0,1).contiguous().view([nwords,self.h * self.d_k]) # [86,8]
        self.attn=p_attn  #[2,86,86]
        
        # 3) "Concat" using a view and apply a final linear. 
       # x = x.transpose(1, 2).contiguous() \
       #      .view(nwords, -1, self.h * self.d_k)
        #print(query.shape)  #[86,2,1,4]
        #print(self.linears[-1](x).unsqueeze(1).shape) #[86,1,8]
        #a=input()
        return self.linears[-1](x).unsqueeze(1)  #[86,1,8]

class tgt_MultiHeadedAttention(nn.Module):    
    def __init__(self, h, d_model, dropout=0):
        "Take in model size and number of heads."
        super(tgt_MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, query, key, value, mask=None):   #q=tgt, k=self, v=self 
        nwords = key.size(0) #nwords==86

        # 1) Do all the linear projections in batch from d_model => h x d_k 
        '''lk=[]
        for i in range(nwords):
          lk.append(query)
        query=torch.stack(lk,0).squeeze(1)
        '''
        query = self.linears[0](query).view(-1,self.h,self.d_k).transpose(0,1)
        key   = self.linears[1](key).view(nwords,-1,self.h,self.d_k).transpose(1,2)
        value = self.linears[2](value).view(nwords,-1,self.h,self.d_k).transpose(1,2)
        
        '''query, key, value = \
            [l(x).view(nwords, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]'''

        #print(key.shape)   [86,2,1,4]
        # 2) Apply attention on all the projected vectors in batch. 
        #x, self.attn = attention(query, key, value, mask=mask, 
        #                         dropout=self.dropout)
        d_k=query.size(-1)
        h=query.size(0)
        key=key.transpose(0,1)  # [2,86,1,4]
        #query  #[2,1,4]?
        key=key.squeeze(2).transpose(1,2)  # [2,4,86]
        scores = torch.matmul(query,key) # [2,1,86]? 
        p_attn = F.softmax(scores, dim = 2) 
        #print(scores.shape) [1,1,806]
        #print(p_attn.shape) [1,1,806]
        #a=input()
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
        value=value.squeeze(2).transpose(0,1)  #[2,86,4]
        
        x=torch.matmul(p_attn, value)  #[2,1,4]     CHECK TWICE!!! ATTN X VALUE  IMPORTANT!!!        
        x=x.transpose(0,1).contiguous().view([1,self.h * self.d_k]) # [1,8]
        self.attn=p_attn  #[2,1,86]
        
        return self.linears[-1](x)  #[1,1,8]

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=13100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2)).type(torch.FloatTensor) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position.type(torch.FloatTensor) * div_term.type(torch.FloatTensor))
        pe[:, 1::2] = torch.cos(position.type(torch.FloatTensor) * div_term.type(torch.FloatTensor))
        #pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        #print(x.size())
        #print(Variable(self.pe[0:x.size(0), :x.size(1)], 
        #                 requires_grad=False).size())
        x = x + Variable(self.pe[0:x.size(0), :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        '''
        norm=self.a_2  
        a=x-mean
        norm=norm* a
        norm=norm/(std + self.eps)
        bias=self.b_2'''
        #return norm+bias
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #print("posff:")
        #print(x.shape)    [86,86,8]
        #print(self.w_2(self.dropout(F.relu(self.w_1(x)))).shape)  [86,2,86,4]
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, dataset):
        np.random.shuffle(dataset)
        loss_total = 0
        for data in dataset:
            loss = self.model(data)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.to('cpu').data.numpy()
        return loss_total

    '''def multigpu_train(self, dataset, device_ids=[0], output_device=None):
        if not device_ids:
          return module(dataset)
        
        if output_device is None:
          output_device = device_ids[0]
        
        replicas = nn.parallel.replicate(self.model, device_ids)
        for data in dataset:
            inputs = nn.parallel.scatter(data, device_ids)
            #print("inputs:")
            #print(inputs[0])
            #print(np.array(inputs).shape)
            #replicas= replicas[:len(inputs)]
            l=nn.parallel.parallel_apply(replicas, inputs)
            loss=nn.parallel.gather(l,output_device)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.to('cpu').data.numpy()
        return loss_total
     '''

class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataset):

        z_list, t_list = [], []
        for data in dataset:
            z, t = self.model(data, train=False)
            z_list.append(z)
            t_list.append(t)

        score_list, label_list = [], []
        for z in z_list:
            score_list.append(z[1])
            label_list.append(np.argmax(z))
        
        auc = roc_auc_score(t_list, score_list)
        precision = precision_score(t_list, label_list)
        recall = recall_score(t_list, label_list)

        return auc, precision, recall

    def result(self, epoch, time, loss, auc_dev,
               auc_test, precision, recall, file_name):
        with open(file_name, 'a') as f:
            result = map(str, [epoch, time, loss, auc_dev,
                               auc_test, precision, recall])
            f.write('\t'.join(result) + '\n')

    def save_model(self, model, file_name):
        torch.save(model.state_dict(), file_name)


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

    (DATASET,dataname, radius, ngram, dim, d_ff, layer_gnn, layer_output,heads, n_encoder,n_decoder,
     lr, lr_decay, decay_interval, weight_decay, iteration,
     setting,dropout) = sys.argv[1:]
    (dim, d_ff, layer_gnn, layer_output, decay_interval,
     iteration,heads, n_encoder,n_decoder,dropout) = map(int, [dim, d_ff, layer_gnn, layer_output,
                            decay_interval, iteration,heads, n_encoder,n_decoder,dropout])
    lr, lr_decay, weight_decay = map(float, [lr, lr_decay, weight_decay])
    if (dim==0):
      print("Unapplicable model dimension! Quit.")
      quit() 
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    dir_input = ('../dataset/' + DATASET + '/input/'
                 'radius' + radius + '_ngram' + ngram +' '+dataname +'/')
    compounds = load_tensor(dir_input + 'compounds', torch.LongTensor)  # [[subgraph type index,...],...]  --JW
    adjacencies = load_tensor(dir_input + 'adjacencies', torch.FloatTensor)
    proteins = load_tensor(dir_input + 'proteins', torch.LongTensor)    # [[word type index,...],...]  --JW
    interactions = load_tensor(dir_input + 'interactions', torch.LongTensor)

    dataset = list(zip(compounds, adjacencies, proteins, interactions))
    dataset = shuffle_dataset(dataset, 1234)
    dataset_train, dataset_ = split_dataset(dataset, 0.8)
    dataset_dev, dataset_test = split_dataset(dataset_, 0.5)
   # print(dataset_dev)
   # print("dataset_dev")
   # print(dataset)
   # print("dataset")
   # print(dataset_train)
   # print("dataset_train")
   # print(dataset_test)
   # print("dataset_test")
    fingerprint_dict = load_pickle(dir_input + 'fingerprint_dict.pickle')  # key: subgraph type, value: subgraph type index --JW
    word_dict = load_pickle(dir_input + 'word_dict.pickle')  # key: [Amino_i:Amino_i+ngram], value: word type index  --JW
    n_fingerprint = len(fingerprint_dict) + 1 # dictionaries are only used for embeddings  --JW
    n_word = len(word_dict) + 1               # CNN&GNN only use keys !!!                  --JW
    #print("n_word")
    #print(n_word)
    torch.manual_seed(1234)
    model = CompoundProteinInteractionPrediction().to(device)
    # this is important for transformer:
    for p in model.parameters():
      if p.dim()>1:
        nn.init.xavier_uniform_(p)
    #model=nn.DataParallel(model)
    trainer = Trainer(model)
    tester = Tester(model)

    file_result = '../output/result/' +dataname+ " "+setting +  ' T.txt'
    with open(file_result, 'w') as f:
        f.write('Epoch\tTime(sec)\tLoss_train\tAUC_dev\t'
                'AUC_test\tPrecision_test\tRecall_test\n')

    file_model = '../output/model/' +dataname+''+ setting

    print('Training...')
    print('Epoch Time(sec) Loss_train AUC_dev '
          'AUC_test Precision_test Recall_test')

    start = timeit.default_timer()
    
    for epoch in range(iteration):

        if (epoch+1) % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay
        
        loss = trainer.train(dataset_train)  # dataset_train: [[compounds,adjacencies,proteins,interactions],...]  --JW
        auc_dev = tester.test(dataset_dev)[0]
        auc_test, precision, recall = tester.test(dataset_test)

        end = timeit.default_timer()
        time = end - start

        tester.result(epoch, time, loss, auc_dev,
                      auc_test, precision, recall, file_result)
        tester.save_model(model, file_model)

        print(epoch, time, loss, auc_dev, auc_test, precision, recall)
