#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import argparse
import numpy as np
import timeit
import torch
import torch.utils.data as data_utils
import math
import pickle



# In[2]:


x_train=np.load('/data/ag_news/processed/x_train.npy')
x_test=np.load('/data/ag_news/processed/x_test.npy')

y_train=np.load('/data/ag_news/processed/y_train.npy')
y_test=np.load('/data/ag_news/processed/y_test.npy')
print(x_train.shape)

x_train=x_train[:,:100]
x_test=x_test[:,:100]


def load_glove_embeddings(path, word2idx, embedding_dim):
    """Loading the glove embeddings"""
    with open(path) as f:
        embeddings = np.zeros((len(word2idx), embedding_dim))
        for line in f.readlines():
            values = line.split()
            word = values[0]
            index = word2idx.get(word)
            if index:
                vector = np.array(values[1:], dtype='float32')
                if vector.shape[-1] != embedding_dim:
                    raise Exception('Dimension not matching.')
                embeddings[index] = vector
        return torch.from_numpy(embeddings).float()

with open('/data/ag_news/processed/word_emb.txt','r') as f:
    vocab=eval(f.read())

#load glove 
# pretrain='glove'
# embedding_dim=300
# if pretrain=='glove':
#     file_path=os.path.join('../../glove.6B','glove.6B.%dd.txt'%(embedding_dim))
#     embedding_weights = load_glove_embeddings(file_path,vocab,embedding_dim)

# embedding_weights.shape

batch_size=64
train_data = data_utils.TensorDataset(torch.from_numpy(x_train).type(torch.LongTensor),torch.from_numpy(y_train).type(torch.LongTensor))

test_data = data_utils.TensorDataset(torch.from_numpy(x_test).type(torch.LongTensor),torch.from_numpy(y_test).type(torch.LongTensor))
train_loader = data_utils.DataLoader(train_data, batch_size, drop_last=False,shuffle=True)

test_loader = data_utils.DataLoader(test_data, batch_size, drop_last=False)


#create Network structure
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# In[5]:

class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule,self).__init__()
        self.model_name=str(type(self))
        
    def load(self,path):
        self.load_state_dict(torch.load(path,map_location=lambda storage, loc: storage.cuda(0)))
        
    def save(self,path=None):
        if path is None:
            raise ValueError('Please specify the saving road!!!')
        torch.save(self.state_dict(),path)
        return path
    


def get_embedding_layer(embedding_weights):
    word_embeddings=nn.Embedding(embedding_weights.size(0),embedding_weights.size(1))
    word_embeddings.weight.data.copy_(embedding_weights)
    word_embeddings.weight.requires_grad=False #not train
    return word_embeddings
class LSTM(BasicModule):
    def __init__(self,num_labels=4,vocab_size=42783,embedding_size=300,embedding_weights=embedding_weights,
                 hidden_size=300,seq_len=100,step=10,label_emb=None):
        super(LSTM,self).__init__()
        self.embedding_size=embedding_size
        self.num_labels=num_labels
        self.hidden_size=hidden_size
        
        if embedding_weights is None:
            self.word_embeddings=nn.Embedding(vocab_size,embedding_size)
        else:
            self.word_embeddings=get_embedding_layer(embedding_weights)
            
        self.lstm=nn.LSTM(input_size=self.embedding_size,hidden_size=self.hidden_size,num_layers=1,batch_first=True,bidirectional=True)
        self.linear1=nn.Linear(2*self.hidden_size,self.hidden_size)
        
        self.pool=nn.AdaptiveMaxPool1d(output_size=int(seq_len//step))
        self.conv=nn.Conv1d(in_channels=seq_len,out_channels=1,kernel_size=1,padding=0)
        
        self.adaptive_factor=nn.Linear(300,1)
        


        label_embedding=torch.FloatTensor(self.num_labels,self.hidden_size)
        if label_emb is None:
            nn.init.xavier_normal_(label_embedding)
        else:
            label_embedding.copy_(label_emb)
        self.label_embedding=nn.Parameter(label_embedding,requires_grad=True)


    def init_hidden(self,batch_size):
        if torch.cuda.is_available():
            return (torch.zeros(2,batch_size,self.hidden_size).cuda(),torch.zeros(2,batch_size,self.hidden_size).cuda())
        else:
            return (torch.zeros(2,batch_size,self.hidden_size),torch.zeros(2,batch_size,self.hidden_size))
                
    def forward(self,x,test=False):
        
        emb=self.word_embeddings(x)
        hidden_state=self.init_hidden(emb.size(0))
        
        output,hidden_state=self.lstm(emb,hidden_state)#[batch,seq,2*hidden]
        output=F.relu(self.linear1(output))#[batch,seq,hidden]
        
        pool_output=self.pool(output.transpose(1,2))#[batch,hidden,seq/10]
        
        adaptive_factor=torch.tanh(self.adaptive_factor(pool_output.transpose(1,2))).squeeze(-1)
        adaptive_factor=F.softmax(adaptive_factor,dim=-1)


        out=torch.matmul(pool_output.transpose(1,2),self.label_embedding.transpose(0,1))#[batch, seq/10 , L]
        channel=out.shape[1]
        
        high_out=self.conv(output).squeeze(1)#[batch,hidden]
        if not test:
            out=out.reshape(-1,self.num_labels)#[batch*C , L]
            out2=torch.matmul(high_out,self.label_embedding.transpose(0,1))#[batch,L]
            return out,out2,channel,adaptive_factor
        else:
            out1=torch.matmul(high_out,self.label_embedding.transpose(0,1))#[batch,L]
            out1=F.softmax(out1,dim=-1)
            
            out2=F.softmax(out,dim=-1)
            out2=torch.sum(out2*adaptive_factor.unsqueeze(2),dim=1)
            return out2,out1,adaptive_factor
        
           
use_cuda=torch.cuda.is_available()
torch.cuda.set_device(0)
torch.manual_seed(1234)
if use_cuda:
    torch.cuda.manual_seed(1234)


# In[111]:


model=LSTM(num_labels=4,vocab_size=42783,embedding_size=300,embedding_weights=None,
           hidden_size=300,seq_len=100,step=10)

model.cuda()



# In[112]:


optimizer=torch.optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=0.001,weight_decay=4e-5)
criterion=torch.nn.CrossEntropyLoss(reduction='none')

epoch=10
best_acc=0.0
pre_acc=0.0
trace_file='./results/ag_news/trace.txt'
for ep in range(1,epoch+1): 
    train_loss=0.0
    train_loss_channel=0.0
    train_loss_top=0.0
    train_acc=0.0
    print("----epoch: %2d---- "%ep)
    model.train()
    for idx,(data,labels) in enumerate(train_loader):
        optimizer.zero_grad()
        if use_cuda:
            data=data.cuda()
            labels_idx=labels.reshape(-1).cuda()
        
        pred1,pred2,channel,factor=model(data)
        
        bs=labels_idx.shape[0]
        labels_idx2=labels_idx.unsqueeze(1).expand(bs,channel).reshape(-1)
        loss1=criterion(pred1,labels_idx2)
        loss1=loss1.reshape(factor.shape)
        loss1=torch.sum(loss1*factor)/bs

        

        loss2=torch.mean(criterion(pred2,labels_idx))
        loss_sum=loss1+loss2

        
        loss_sum.backward()
        optimizer.step()
        
        train_loss+=float(loss_sum)
        train_loss_channel+=float(loss1)
        train_loss_top+=float(loss2)
#         #compute acc
        train_acc+=float(torch.eq(torch.max(pred2.data,dim=1)[1],labels_idx).float().mean())

    batch_num=idx+1
    train_loss/=batch_num
    train_loss_channel/=batch_num
    train_loss_top/=batch_num
    train_acc/=batch_num

    
    print("epoch %2d training ends : avg_loss = %.4f "%(ep,train_loss))

    
    print("Begin validation")
    test_loss=0.0
    test_acc=0.0
    model.eval()

    for test_id,(data,labels) in enumerate(test_loader):
#         labels_idx=torch.max(labels,dim=1)[1].view(-1)
        if use_cuda:
            data=data.cuda()
            labels_idx=labels.view(-1).cuda()
        pred1,pred2,factor=model(data,test=True)
#         print(pred1.shape)
        pred=pred1+pred2
#         loss=criterion(pred,labels_idx)
        
        test_acc+=float(torch.eq(torch.max(pred.data,dim=1)[1],labels_idx).float().mean())
#         test_loss+=float(loss)
        
    batch_num=test_id+1
#     test_loss/=batch_num
    test_acc/=batch_num

    print("epoch %2d testing ends :avg_acc = %.4f"%(ep,test_acc))


    with open(trace_file,'a') as f:
        f.write('epoch:{:2d} testing ends：avg_acc:{:.4f}'.format(ep,test_acc))
        f.write('\n')
    if test_acc>best_acc:
        best_acc=test_acc
        name=model.save(path='./results/ag_news/best_model.m')
        print("save",name)

    if  test_acc<pre_acc:
        for param_group in optimizer.param_groups:
            param_group['lr']=0.0001
    pre_acc=test_acc