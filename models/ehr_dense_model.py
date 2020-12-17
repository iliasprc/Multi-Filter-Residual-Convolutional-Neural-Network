import torch
import torch.nn as nn
from torch import Tensor
from typing import Any, List, Tuple
import torch.nn.functional as F
from collections import OrderedDict
from models.attn import Attn
import torch.nn.init
from elmo.elmo import Elmo
import json
from utils.utils import build_pretrain_embedding, load_embeddings
from math import floor
import math


class WordRep(nn.Module):
    def __init__(self, args, Y, dicts):
        super(WordRep, self).__init__()

        self.gpu = args.gpu

        if args.embed_file:
            print("loading pretrained embeddings from {}".format(args.embed_file))
            if args.use_ext_emb:
                pretrain_word_embedding, pretrain_emb_dim = build_pretrain_embedding(args.embed_file, dicts['w2ind'],
                                                                                     True)
                W = torch.from_numpy(pretrain_word_embedding)
            else:
                W = torch.Tensor(load_embeddings(args.embed_file))

            self.embed = nn.Embedding(W.size()[0], W.size()[1], padding_idx=0)
            self.embed.weight.data = W.clone()
        else:
            # add 2 to include UNK and PAD
            self.embed = nn.Embedding(len(dicts['w2ind']) + 2, args.embed_size, padding_idx=0)
        self.feature_size = self.embed.embedding_dim

        self.use_elmo = args.use_elmo
        if self.use_elmo:
            self.elmo = Elmo(args.elmo_options_file, args.elmo_weight_file, 1, requires_grad=args.elmo_tune,
                             dropout=args.elmo_dropout, gamma=args.elmo_gamma)
            with open(args.elmo_options_file, 'r') as fin:
                _options = json.load(fin)
            self.feature_size += _options['lstm']['projection_dim'] * 2

        self.embed_drop = nn.Dropout(p=args.dropout)

        self.conv_dict = {1: [self.feature_size, args.num_filter_maps],
                     2: [self.feature_size, 100, args.num_filter_maps],
                     3: [self.feature_size, 150, 100, args.num_filter_maps],
                     4: [self.feature_size, 200, 150, 100, args.num_filter_maps]
                     }


    def forward(self, x, target, text_inputs):

        features = [self.embed(x)]

        if self.use_elmo:
            elmo_outputs = self.elmo(text_inputs)
            elmo_outputs = elmo_outputs['elmo_representations'][0]
            features.append(elmo_outputs)

        x = torch.cat(features, dim=2)

        x = self.embed_drop(x)
        return x


class DenseBlock(nn.Module):
    def __init__(self,input_size,num_filters,kernel_size):
        super(DenseBlock,self).__init__()

        self.conv1= nn.Conv1d(input_size, num_filters, kernel_size=kernel_size, stride=1,padding=1)
        self.relu1= nn.ReLU()
        self.norm1= nn.BatchNorm1d(num_filters)
    def forward(self, x):
        out = self.norm1(self.relu1(self.conv1(x)))
        return out
class OutputLayer(nn.Module):
    def __init__(self, args, Y, dicts, input_size):
        super(OutputLayer, self).__init__()

        self.U = nn.Linear(input_size, Y)
        torch.nn.init.xavier_uniform(self.U.weight)


        self.final = nn.Linear(input_size, Y)
        torch.nn.init.xavier_uniform(self.final.weight)

        self.loss_function = nn.BCEWithLogitsLoss()



    def forward(self, x, target, text_inputs):
        #print(f'Inside out layer {x.shape} x.transpose {x.transpose(1, 2).shape}')
        alpha = F.softmax(self.U.weight.matmul(x.transpose(1, 2)), dim=2)
        #print(f'alpha {alpha.shape} x {x.shape} x.transpose {x.transpose(1, 2).shape}')
        m = alpha.matmul(x)
        #print(f'alpha {alpha.shape} x {x.shape} x.transpose {x.transpose(1, 2).shape} self.final.weight.mul(m) {self.final.weight.mul(m).sum(dim=2).shape}')
        y = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)
        #y = self.final(m.mean(dim=1))

        loss = self.loss_function(y, target)
        return y, loss
class Ehr_Dense_CNN(nn.Module):
    def __init__(self, args, Y, dicts,K=7):
        super(MultiScaleAtt, self).__init__()
        self.word_rep = WordRep(args, Y, dicts)
        filters = [100]
        dc =200
        for i in range(2,K+1):
            filters += [dc]
            print(filters,sum(filters[:-1]))
            self.add_module(f"block{i-2}",DenseBlock(sum(filters[:-1]),filters[i-1],3))
        self.output_layer = OutputLayer(args, Y, dicts,dc)
    def forward(self, x, target, text_inputs):
        x = self.word_rep(x, target, text_inputs)
        x = x.transpose(1, 2)
        x1 = self.block0(x)
        x2 = self.block1(torch.cat((x1,x),dim=1))

        x3 = self.block2(torch.cat((x2,x1,x),dim=1))
        x4 = self.block3(torch.cat((x3,x2,x1,x),dim=1))
        x5 = self.block4(torch.cat((x4,x3,x2,x1,x),dim=1))
        x6 = self.block5(torch.cat((x5,x4, x3, x2, x1, x), dim=1))
        # s1 = torch.sum(x1,dim=1)
        # s2 = torch.sum(x2, dim=1)
        # s3 = torch.sum(x3, dim=1)
        # s4 = torch.sum(x4, dim=1)
        # s5 = torch.sum(x5, dim=1)
        #print(x5.shape)
        x6 = x6.transpose(1, 2)
        # out1 , loss1= self.output_layer(x1,target, text_inputs)
        # out2, loss2 = self.output_layer(x2, target, text_inputs)
        # out3, loss3 = self.output_layer(x3, target, text_inputs)
        # out, loss = self.output_layer(x4, target, text_inputs)
        # out, loss = self.output_layer(x5, target, text_inputs)
        out, loss = self.output_layer(x6, target, text_inputs)
        return out,loss




class MultiScaleAtt(nn.Module):
    def __init__(self, args, Y, dicts,K=7):
        super(MultiScaleAtt, self).__init__()
        self.word_rep = WordRep(args, Y, dicts)
        self.att1 = Attn('',100)
        filters = [100]
        dc =200
        for i in range(2,K+1):
            filters += [dc]
            print(filters,sum(filters[:-1]))
            self.add_module(f"block{i-2}",DenseBlock(sum(filters[:-1]),filters[i-1],3))
        self.att = Attn('bmm',200)
        #self.output_layer = OutputLayer(args, Y, dicts,dc)
        self.output_layer = nn.Linear( dc,Y)
        self.loss_function = nn.BCEWithLogitsLoss()
    def forward(self, x, target, text_inputs):
        x = self.word_rep(x, target, text_inputs)
        x = self.att1(x)
        x = x.transpose(1, 2)
        #print(x.shape)

        x1 = self.block0(x)
        x2 = self.block1(torch.cat((x1,x),dim=1))

        x3 = self.block2(torch.cat((x2,x1,x),dim=1))
        x4 = self.block3(torch.cat((x3,x2,x1,x),dim=1))
        x5 = self.block4(torch.cat((x4,x3,x2,x1,x),dim=1))
        x6 = self.block5(torch.cat((x5,x4, x3, x2, x1, x), dim=1))

        # s1 = torch.sum(x1,dim=1)
        # s2 = torch.sum(x2, dim=1)
        # s3 = torch.sum(x3, dim=1)
        # s4 = torch.sum(x4, dim=1)
        # s5 = torch.sum(x5, dim=1)
        #print(x6.shape)
        #x_cat = torch.stack((x1,x2,x3,x4,x5,x6),dim=-1)
        #print(x_cat.shape)
        x6 = x6.permute(0,2,1)
        c = self.att(x6)# +self.att(x1.permute(0,2,1)) + self.att(x2.permute(0,2,1)) + self.att(x3.permute(0,2,1)) + self.att(x4.permute(0,2,1)) + self.att(x5.permute(0,2,1))#,x6,x6)
       # x_cat = x_cat.permute(1,0,2)

        #print(x_cat.shape)
        # out1 , loss1= self.output_layer(x1,target, text_inputs)
        # out2, loss2 = self.output_layer(x2, target, text_inputs)
        # out3, loss3 = self.output_layer(x3, target, text_inputs)
        # out, loss = self.output_layer(x4, target, text_inputs)
        # out, loss = self.output_layer(x5, target, text_inputs)
        out = self.output_layer(c)
        loss = self.loss_function(out,target)
        return out,loss

class MultiScaleAttention(nn.Module):
    def __init__(self,K=6):
        super(MultiScaleAttention,self).__init__()
        self.mlp = nn.Sequential(nn.Linear(K,1),nn.Softmax(dim=-1))

    def forward(self,x):
        s = torch.sum(x,dim=1)
        print(s.shape)
        a = self.mlp(s)
        print(f"a {a.shape} s {s.shape} x {x.shape}")
        x_attn = torch.sum(torch.matmul(a,torch.sum(x,dim=-1)),dim=-1)
        print(x_attn.shape)

        return x_attn


