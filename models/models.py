
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_ as xavier_uniform
from elmo.elmo import Elmo
import json
from utils.utils import build_pretrain_embedding, load_embeddings
from math import floor
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)



    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)  # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)

        output = self.out(concat)

        return output


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output
class MultiHeadResCNN(nn.Module):

    def __init__(self, args, Y, dicts):
        super(MultiHeadResCNN, self).__init__()

        self.word_rep = WordRep(args, Y, dicts)

        self.conv = nn.ModuleList()
        filter_sizes = args.filter_size.split(',')

        self.filter_num = len(filter_sizes)
        for filter_size in filter_sizes:
            filter_size = int(filter_size)
            one_channel = nn.ModuleList()
            tmp = nn.Conv1d(self.word_rep.feature_size, self.word_rep.feature_size, kernel_size=filter_size,
                            padding=int(floor(filter_size / 2)))
            xavier_uniform(tmp.weight)
            one_channel.add_module('baseconv', tmp)

            conv_dimension = self.word_rep.conv_dict[args.conv_layer]
            for idx in range(args.conv_layer):
                tmp = ResidualBlock(conv_dimension[idx], conv_dimension[idx + 1], filter_size, 1, True,
                                    args.dropout)
                one_channel.add_module('resconv-{}'.format(idx), tmp)

            self.conv.add_module('channel-{}'.format(filter_size), one_channel)
        self.attention = MultiHeadAttention(10,self.filter_num * args.num_filter_maps)
        self.output_layer = nn.Linear(self.filter_num * args.num_filter_maps,Y)

        self.loss = nn.BCEWithLogitsLoss()
    def forward(self, x, target, text_inputs):
        #print(x.shape,text_inputs.shape,target.shape)
        x = self.word_rep(x, target, text_inputs)
        #print(x.shape)
        x = x.transpose(1, 2)

        conv_result = []
        for conv in self.conv:
            tmp = x
            for idx, md in enumerate(conv):
                if idx == 0:
                    tmp = torch.tanh(md(tmp))
                else:
                    tmp = md(tmp)
            tmp = tmp.transpose(1, 2)
            conv_result.append(tmp)
        x = torch.cat(conv_result, dim=2)
        x_mh = self.attention(x,x,x)
        #print(embeds.shape)
        #x_embeds = x.mean(dim=1)
        #print(x_embeds.shape)
        #loss_emb = self.sim(x_embeds,embeds,torch.Tensor([1]).cuda())
        #loss_emb = ((x_embeds-embeds)**2).mean()
        y= self.output_layer(x_mh)
        #print(f' before output {x.shape} out -> {y.shape} target {target.shape} ')
        return y, self.loss(y,target)#+loss_emb

    def freeze_net(self):
        for p in self.word_rep.embed.parameters():
            p.requires_grad = False
class Mapper(nn.Module):
    def __init__(self,hidden_size=300,n_classes=50):
        super(Mapper, self).__init__()

        self.embed = nn.Linear(n_classes,hidden_size)

        xavier_uniform(self.embed.weight)
    def forward(self,x):
        return self.embed(x)



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

class OutputLayer(nn.Module):
    def __init__(self, args, Y, dicts, input_size):
        super(OutputLayer, self).__init__()

        self.U = nn.Linear(input_size, Y)
        xavier_uniform(self.U.weight)


        self.final = nn.Linear(input_size, Y)
        xavier_uniform(self.final.weight)

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


class CNN(nn.Module):
    def __init__(self, args, Y, dicts):
        super(CNN, self).__init__()

        self.word_rep = WordRep(args, Y, dicts)

        filter_size = int(args.filter_size)


        self.conv = nn.Conv1d(self.word_rep.feature_size, args.num_filter_maps, kernel_size=filter_size,
                                  padding=int(floor(filter_size / 2)))
        xavier_uniform(self.conv.weight)

        self.output_layer = OutputLayer(args, Y, dicts, args.num_filter_maps)


    def forward(self, x, target, text_inputs):

        x = self.word_rep(x, target, text_inputs)

        x = x.transpose(1, 2)

        x = torch.tanh(self.conv(x).transpose(1, 2))

        y, loss = self.output_layer(x, target, text_inputs)
        return y, loss

    def freeze_net(self):
        for p in self.word_rep.embed.parameters():
            p.requires_grad = False


class MultiCNN(nn.Module):
    def __init__(self, args, Y, dicts):
        super(MultiCNN, self).__init__()

        self.word_rep = WordRep(args, Y, dicts)

        if args.filter_size.find(',') == -1:
            self.filter_num = 1
            filter_size = int(args.filter_size)
            self.conv = nn.Conv1d(self.word_rep.feature_size, args.num_filter_maps, kernel_size=filter_size,
                                  padding=int(floor(filter_size / 2)))
            xavier_uniform(self.conv.weight)
        else:
            filter_sizes = args.filter_size.split(',')
            self.filter_num = len(filter_sizes)
            self.conv = nn.ModuleList()
            for filter_size in filter_sizes:
                filter_size = int(filter_size)
                tmp = nn.Conv1d(self.word_rep.feature_size, args.num_filter_maps, kernel_size=filter_size,
                                      padding=int(floor(filter_size / 2)))
                xavier_uniform(tmp.weight)
                self.conv.add_module('conv-{}'.format(filter_size), tmp)

        self.output_layer = OutputLayer(args, Y, dicts, self.filter_num * args.num_filter_maps)



    def forward(self, x, target, text_inputs):

        x = self.word_rep(x, target, text_inputs)

        x = x.transpose(1, 2)

        if self.filter_num == 1:
            x = torch.tanh(self.conv(x).transpose(1, 2))
        else:
            conv_result = []
            for tmp in self.conv:
                conv_result.append(torch.tanh(tmp(x).transpose(1, 2)))
            x = torch.cat(conv_result, dim=2)

        y, loss = self.output_layer(x, target, text_inputs)

        return y, loss

    def freeze_net(self):
        for p in self.word_rep.embed.parameters():
            p.requires_grad = False


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, stride, use_res, dropout):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv1d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=int(floor(kernel_size / 2)), bias=False),
            nn.BatchNorm1d(outchannel),
            nn.Tanh(),
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

class ResCNN(nn.Module):

    def __init__(self, args, Y, dicts):
        super(ResCNN, self).__init__()

        self.word_rep = WordRep(args, Y, dicts)

        self.conv = nn.ModuleList()
        conv_dimension = self.word_rep.conv_dict[args.conv_layer]
        for idx in range(args.conv_layer):
            tmp = ResidualBlock(conv_dimension[idx], conv_dimension[idx + 1], int(args.filter_size), 1, True, args.dropout)
            self.conv.add_module('conv-{}'.format(idx), tmp)

        self.output_layer = OutputLayer(args, Y, dicts, args.num_filter_maps)


    def forward(self, x, target, text_inputs):

        x = self.word_rep(x, target, text_inputs)

        x = x.transpose(1, 2)

        for conv in self.conv:
            x = conv(x)
        x = x.transpose(1, 2)

        y, loss = self.output_layer(x, target, text_inputs)

        return y, loss

    def freeze_net(self):
        for p in self.word_rep.embed.parameters():
            p.requires_grad = False


class MultiResCNN(nn.Module):

    def __init__(self, args, Y, dicts):
        super(MultiResCNN, self).__init__()

        self.word_rep = WordRep(args, Y, dicts)

        self.conv = nn.ModuleList()
        filter_sizes = args.filter_size.split(',')

        self.filter_num = len(filter_sizes)
        for filter_size in filter_sizes:
            filter_size = int(filter_size)
            one_channel = nn.ModuleList()
            tmp = nn.Conv1d(self.word_rep.feature_size, self.word_rep.feature_size, kernel_size=filter_size,
                            padding=int(floor(filter_size / 2)))
            xavier_uniform(tmp.weight)
            one_channel.add_module('baseconv', tmp)

            conv_dimension = self.word_rep.conv_dict[args.conv_layer]
            for idx in range(args.conv_layer):
                tmp = ResidualBlock(conv_dimension[idx], conv_dimension[idx + 1], filter_size, 1, True,
                                    args.dropout)
                one_channel.add_module('resconv-{}'.format(idx), tmp)

            self.conv.add_module('channel-{}'.format(filter_size), one_channel)
        self.mapper = Mapper(self.filter_num * args.num_filter_maps, Y)
        self.output_layer = OutputLayer(args, Y, dicts, self.filter_num * args.num_filter_maps)

        self.sim = nn.CosineEmbeddingLoss()
    def forward(self, x, target, text_inputs):
        #print(x.shape,text_inputs.shape,target.shape)
        x = self.word_rep(x, target, text_inputs)
        #print(x.shape)
        x = x.transpose(1, 2)

        conv_result = []
        for conv in self.conv:
            tmp = x
            for idx, md in enumerate(conv):
                if idx == 0:
                    tmp = torch.tanh(md(tmp))
                else:
                    tmp = md(tmp)
            tmp = tmp.transpose(1, 2)
            conv_result.append(tmp)
        x = torch.cat(conv_result, dim=2)
        embeds = self.mapper(target)
        #print(embeds.shape)
        #x_embeds = x.mean(dim=1)
        #print(x_embeds.shape)
        #loss_emb = self.sim(x_embeds,embeds,torch.Tensor([1]).cuda())
        #loss_emb = ((x_embeds-embeds)**2).mean()
        y, loss = self.output_layer(x, target, text_inputs)
        #print(f' before output {x.shape} out -> {y.shape} target {target.shape} ')
        return y, loss#+loss_emb

    def freeze_net(self):
        for p in self.word_rep.embed.parameters():
            p.requires_grad = False

import os
from pytorch_pretrained_bert.modeling import BertLayerNorm
from pytorch_pretrained_bert import BertModel, BertConfig
class Bert_seq_cls(nn.Module):

    def __init__(self, args, Y):
        super(Bert_seq_cls, self).__init__()

        print("loading pretrained bert from {}".format(args.bert_dir))
        config_file = os.path.join(args.bert_dir, 'bert_config.json')
        self.config = BertConfig.from_json_file(config_file)
        print("Model config {}".format(self.config))
        self.bert = BertModel.from_pretrained(args.bert_dir)

        self.dim_reduction = nn.Linear(self.config.hidden_size, args.num_filter_maps)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(args.num_filter_maps, Y)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, target):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        x = self.dim_reduction(pooled_output)
        x = self.dropout(x)
        y = self.classifier(x)

        loss = F.binary_cross_entropy_with_logits(y, target)
        return y, loss

    def init_bert_weights(self, module):

        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def freeze_net(self):
        pass


def pick_model(args, dicts):
    Y = len(dicts['ind2c'])
    if args.model == 'CNN':
        model = CNN(args, Y, dicts)
    elif args.model== 'MultiCNN':
        model = MultiCNN(args, Y, dicts)
    elif args.model == 'ResCNN':
        model = ResCNN(args, Y, dicts)
    elif args.model == 'MultiResCNN':
        model = MultiResCNN(args, Y, dicts)
    elif args.model =='MultiHeadCNN':
        model = MultiHeadResCNN(args, Y, dicts)
    elif args.model == 'bert_seq_cls':
        model = Bert_seq_cls(args, Y)
    else:
        raise RuntimeError("wrong model name")

    if args.test_model:
        sd = torch.load(args.test_model)
        model.load_state_dict(sd)
    if args.gpu >= 0:
        model.cuda(args.gpu)
    return model
