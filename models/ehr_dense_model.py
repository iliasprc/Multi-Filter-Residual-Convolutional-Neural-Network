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
        self.relu1= nn.LeakyReLU(0.2
                                 )
        self.norm1= nn.BatchNorm1d(num_filters)
        self.drop = nn.Dropout(0.1)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        #self.norm1
        out = self.drop(self.norm1(self.relu1(self.conv1(x))))
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
        super(Ehr_Dense_CNN, self).__init__()
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




class AttDense(nn.Module):
    def __init__(self, args, Y, dicts,K=7,attn = 'bandanau'):
        super(AttDense, self).__init__()
        self.word_rep = WordRep(args, Y, dicts)
        self.att1 = Attn('',100)
        filters = [100]
        dc =200
        self.attn = attn
        if attn == 'bandanau':
            for i in range(2, K + 1):
                filters += [dc]
                print(filters, sum(filters[:-1]))
                self.add_module(f"block{i - 2}", DenseBlock(sum(filters[:-1]), filters[i - 1], 3))
                self.add_module(f"U{i - 2}", Attn('bmm', dc))
        else:
            for i in range(2,K+1):
                filters += [dc]
                print(filters,sum(filters[:-1]))
                self.add_module(f"block{i-2}",DenseBlock(sum(filters[:-1]),filters[i-1],3))
                self.add_module(f"U{i-2}",nn.Linear(dc,Y))


        #self.att = Attn('bmm',200)

        #self.output_layer = OutputLayer(args, Y, dicts,dc)
        self.output_layer = nn.Linear( dc,Y)
        self.loss_function = nn.BCEWithLogitsLoss()
    def forward(self, x, target, text_inputs):
        x = self.word_rep(x, target, text_inputs)
        #x = self.att1(x)
        x = x.transpose(1, 2)
        #print(x.shape)

        x1 = self.block0(x)
        x2 = self.block1(torch.cat((x1,x),dim=1))

        x3 = self.block2(torch.cat((x2,x1,x),dim=1))
        x4 = self.block3(torch.cat((x3,x2,x1,x),dim=1))
        x5 = self.block4(torch.cat((x4,x3,x2,x1,x),dim=1))
        x6 = self.block5(torch.cat((x5,x4, x3, x2, x1, x), dim=1))
        if self.attn == 'bandanau':
            alpha1 = self.U0(x1.transpose(1,2))
            # print(alpha1.shape,x1.shape)
            alpha2 = self.U0(x2.transpose(1,2))
            alpha3 = self.U0(x3.transpose(1,2))
            alpha4 = self.U0(x4.transpose(1,2))
            alpha5 = self.U0(x5.transpose(1,2))
            alpha6 = self.U0(x6.transpose(1,2))
            #print(alpha1.shape)
            y = self.output_layer((alpha1+alpha2+alpha3+alpha4+alpha5+alpha6))
        else:
            alpha1 = F.softmax(self.U0.weight.matmul(x1), dim=2).matmul(x1.transpose(1,2))
            #print(alpha1.shape,x1.shape)
            alpha2 = (F.softmax(self.U1.weight.matmul(x2), dim=2)).matmul(x2.transpose(1,2))
            alpha3 = (F.softmax(self.U2.weight.matmul(x3), dim=2)).matmul(x3.transpose(1,2))
            alpha4 = (F.softmax(self.U3.weight.matmul(x4), dim=2)).matmul(x4.transpose(1,2))
            alpha5 = (F.softmax(self.U4.weight.matmul(x5), dim=2)).matmul(x5.transpose(1,2))
            alpha6 = (F.softmax(self.U5.weight.matmul(x6), dim=2)).matmul(x6.transpose(1,2))

            # s1 = torch.sum(x1,dim=1)
        # s2 = torch.sum(x2, dim=1)
        # s3 = torch.sum(x3, dim=1)
        # s4 = torch.sum(x4, dim=1)
        # s5 = torch.sum(x5, dim=1)
        #print(x6.shape)
        #x_cat = torch.stack((x1,x2,x3,x4,x5,x6),dim=-1)
        #print(x_cat.shape)
        #x6 = (x6).permute(0,2,1)
        #c = self.att(x6) +self.att(x1.permute(0,2,1)) + self.att(x2.permute(0,2,1)) + self.att(x3.permute(0,2,1)) + self.att(x4.permute(0,2,1)) + self.att(x5.permute(0,2,1))#,x6,x6)
       # x_cat = x_cat.permute(1,0,2)

        #print(x_cat.shape)
        # out1 , loss1= self.output_layer(x1,target, text_inputs)
        # out2, loss2 = self.output_layer(x2, target, text_inputs)
        # out3, loss3 = self.output_layer(x3, target, text_inputs)
        # out, loss = self.output_layer(x4, target, text_inputs)
        # out, loss = self.output_layer(x5, target, text_inputs)
            print(alpha1.shape)
            y = self.output_layer.weight.mul((alpha1+alpha2+alpha3+alpha4+alpha5+alpha6)).sum(dim=2).add(self.output_layer.bias)
        #out = self.output_layer(c)
        loss = self.loss_function(y,target)
        return y,loss



class MultiScaleAttDense(nn.Module):
    def __init__(self, args, Y, dicts,K=7,attn = 'bandanau'):
        super(MultiScaleAttDense, self).__init__()
        self.word_rep = WordRep(args, Y, dicts)
        self.att1 = Attn('',100)
        filters = [100]
        dc =200
        self.attn = attn
        if attn == 'bandanau':
            for i in range(2, K + 1):
                filters += [dc]
                print(filters, sum(filters[:-1]))
                self.add_module(f"block{i - 2}", DenseBlock(sum(filters[:-1]), filters[i - 1], 3))
            self.multi_att = MultiScaleAttention(K-1)
            self.U = nn.Linear(dc,Y)
        else:
            for i in range(2,K+1):
                filters += [dc]
                print(filters,sum(filters[:-1]))
                self.add_module(f"block{i-2}",DenseBlock(sum(filters[:-1]),filters[i-1],3))
                self.add_module(f"U{i-2}",nn.Linear(dc,Y))



        self.output_layer = nn.Linear( dc,Y)
        self.loss_function = nn.BCEWithLogitsLoss()
    def forward(self, x, target, text_inputs):
        x = self.word_rep(x, target, text_inputs)
        #x = self.att1(x)
        x = x.transpose(1, 2)
        #print(x.shape)

        x1 = self.block0(x)
        x2 = self.block1(torch.cat((x1,x),dim=1))

        x3 = self.block2(torch.cat((x2,x1,x),dim=1))
        x4 = self.block3(torch.cat((x3,x2,x1,x),dim=1))
        x5 = self.block4(torch.cat((x4,x3,x2,x1,x),dim=1))
        x6 = self.block5(torch.cat((x5,x4, x3, x2, x1, x), dim=1))
        if self.attn == 'bandanau':
            concat = torch.stack((x1,x2,x3,x4,x5,x6),dim=-1)
            x_attn = self.multi_att(concat)#.transpose(1,2)
            #print(self.output_layer.weight.shape,x_attn.shape)
            alpha = F.softmax(self.U.weight.matmul(x_attn.transpose(1,2)), dim=2).matmul(x_attn)
            #print(f"alpha {alpha.shape}")
            y =  self.output_layer.weight.mul(alpha).sum(dim=2).add(self.output_layer.bias)
            #print(f"y {y.shape}")
        else:
            alpha1 = F.softmax(self.U0.weight.matmul(x1), dim=2).matmul(x1.transpose(1,2))
            #print(alpha1.shape,x1.shape)
            alpha2 = (F.softmax(self.U1.weight.matmul(x2), dim=2)).matmul(x2.transpose(1,2))
            alpha3 = (F.softmax(self.U2.weight.matmul(x3), dim=2)).matmul(x3.transpose(1,2))
            alpha4 = (F.softmax(self.U3.weight.matmul(x4), dim=2)).matmul(x4.transpose(1,2))
            alpha5 = (F.softmax(self.U4.weight.matmul(x5), dim=2)).matmul(x5.transpose(1,2))
            alpha6 = (F.softmax(self.U5.weight.matmul(x6), dim=2)).matmul(x6.transpose(1,2))

            # s1 = torch.sum(x1,dim=1)
        # s2 = torch.sum(x2, dim=1)
        # s3 = torch.sum(x3, dim=1)
        # s4 = torch.sum(x4, dim=1)
        # s5 = torch.sum(x5, dim=1)
        #print(x6.shape)
        #x_cat = torch.stack((x1,x2,x3,x4,x5,x6),dim=-1)
        #print(x_cat.shape)
        #x6 = (x6).permute(0,2,1)
        #c = self.att(x6) +self.att(x1.permute(0,2,1)) + self.att(x2.permute(0,2,1)) + self.att(x3.permute(0,2,1)) + self.att(x4.permute(0,2,1)) + self.att(x5.permute(0,2,1))#,x6,x6)
       # x_cat = x_cat.permute(1,0,2)

        #print(x_cat.shape)
        # out1 , loss1= self.output_layer(x1,target, text_inputs)
        # out2, loss2 = self.output_layer(x2, target, text_inputs)
        # out3, loss3 = self.output_layer(x3, target, text_inputs)
        # out, loss = self.output_layer(x4, target, text_inputs)
        # out, loss = self.output_layer(x5, target, text_inputs)
            print(alpha1.shape)
            y = self.output_layer.weight.mul((alpha1+alpha2+alpha3+alpha4+alpha5+alpha6)).sum(dim=2).add(self.output_layer.bias)
        #out = self.output_layer(c)
        loss = self.loss_function(y,target)
        return y,loss





class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, stride, use_res, dropout):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv1d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=int(floor(kernel_size / 2)), bias=False),
            nn.BatchNorm1d(outchannel),
            nn.ReLU(),
            nn.Conv1d(outchannel, outchannel, kernel_size=kernel_size, stride=1, padding=int(floor(kernel_size / 2)), bias=False),
            nn.BatchNorm1d(outchannel)
        )

        self.use_res = use_res
        if self.use_res:
            self.shortcut = nn.Sequential(
                        nn.Conv1d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm1d(outchannel)
                    )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        out = self.left(x)
        if self.use_res:
            out += self.shortcut(x)
        out = torch.tanh(out)
        out = self.dropout(out)
        return out

class ResDenseCNN(nn.Module):
    def __init__(self,args, Y, dicts):
        super(ResDenseCNN,self).__init__()
        self.word_rep = WordRep(args, Y, dicts)
        K = 6
        kernels = [3,5,9,15,19,25]
        filters = [100]
        dc = 100
        for i in range(2, K + 2):
            filters += [dc]
            print(filters, sum(filters[:-1]))
            self.add_module(f"block{i - 2}", ResidualBlock(sum(filters[:-1]), filters[i - 1], kernels[i-2], 1, True,
                                    0.1))
        self.U = nn.Linear(dc, Y)
        self.output_layer = nn.Linear( dc,Y)
        self.loss_function = nn.BCEWithLogitsLoss()
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    def forward(self, x, target, text_inputs):
        x = self.word_rep(x, target, text_inputs)
        # x = self.att1(x)
        x = x.transpose(1, 2)
        # print(x.shape)

        x1 = self.block0(x)
        x2 = self.block1(torch.cat((x1, x), dim=1))

        x3 = self.block2(torch.cat((x2, x1, x), dim=1))
        x4 = self.block3(torch.cat((x3, x2, x1, x), dim=1))
        x5 = self.block4(torch.cat((x4, x3, x2, x1, x), dim=1))
        x6 = self.block5(torch.cat((x5, x4, x3, x2, x1, x), dim=1))
        #print(x6.shape)
        alpha6 = (F.softmax(self.U.weight.matmul(x6), dim=2)).matmul(x6.transpose(1,2))

        # s1 = torch.sum(x1,dim=1)
        # s2 = torch.sum(x2, dim=1)
        # s3 = torch.sum(x3, dim=1)
        # s4 = torch.sum(x4, dim=1)
        # s5 = torch.sum(x5, dim=1)
        #print(x6.shape)
        #x_cat = torch.stack((x1,x2,x3,x4,x5,x6),dim=-1)
        #print(x_cat.shape)
        #x6 = (x6).permute(0,2,1)
        #c = self.att(x6) +self.att(x1.permute(0,2,1)) + self.att(x2.permute(0,2,1)) + self.att(x3.permute(0,2,1)) + self.att(x4.permute(0,2,1)) + self.att(x5.permute(0,2,1))#,x6,x6)
       # x_cat = x_cat.permute(1,0,2)

        #print(x_cat.shape)
        # out1 , loss1= self.output_layer(x1,target, text_inputs)
        # out2, loss2 = self.output_layer(x2, target, text_inputs)
        # out3, loss3 = self.output_layer(x3, target, text_inputs)
        # out, loss = self.output_layer(x4, target, text_inputs)
        # out, loss = self.output_layer(x5, target, text_inputs)
        y = self.output_layer.weight.mul((alpha6)).mean(dim=2).add(self.output_layer.bias)
        #out = self.output_layer(c)
        loss = self.loss_function(y,target)
        return y,loss

class MultiScaleAttention(nn.Module):
    def __init__(self,K=6):
        super(MultiScaleAttention,self).__init__()
        self.mlp = nn.Linear(K,K)

    def forward(self,x):
        #print(f"x {x.shape}")
        # X shape Batch x NumWords X Filters X Layers
        s = torch.sum(x,dim=1)
        #print(f"s {s.shape}")
        # S shape Batch x NumWords X  Layers
        #print(s.shape)
        a = F.softmax(self.mlp(s),dim=-1).unsqueeze(1)
        # A shape Batch x NumWords X  1
        # Softmax
        #print(f"a {a.shape} s {s.shape} x {x.shape}")
        x_attn = torch.sum(a*x,dim=-1).transpose(1,2)
        #print(f"x attn {x_attn.shape}")

        return x_attn


