
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_ as xavier_uniform
from elmo.elmo import Elmo
import json
from utils.utils import build_pretrain_embedding, load_embeddings
from math import floor
from models.dense_models import densenet100,densenet121
from models.ehr_dense_model import DenseBlock
from models.attention import AttnDecoderRNN,Decoder
class Mapper(nn.Module):
    def __init__(self,hidden_size=300,n_classes=50):
        super(Mapper, self).__init__()

        self.embed = nn.Linear(n_classes,hidden_size)

        xavier_uniform(self.embed.weight)
    def forward(self,x):
        return self.embed(x)

def get_kl_loss(mean, logvar):
    """
    KL DIV LOSS SECTION
        Exercise2-LOSSES-(ii): Implement the KL Divergence Loss
        Input: mean, logvar of latent space
        Returns: KL Loss
    """
    ###Solution
    loss_kl = (-0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()))
    return loss_kl

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



        self.final = nn.Linear(input_size, Y)
        xavier_uniform(self.final.weight)

        self.loss_function = nn.BCEWithLogitsLoss()



    def forward(self, x, target, text_inputs):
        #print(f'Inside out layer {x.shape} x.transpose ')
#        alpha = F.softmax(self.U.weight.matmul(x.transpose(1, 2)), dim=2)
        # #print(f'alpha {alpha.shape} x {x.shape} x.transpose {x.transpose(1, 2).shape}')
        # m = alpha.matmul(x)
        # #print(f'alpha {alpha.shape} x {x.shape} x.transpose {x.transpose(1, 2).shape} self.final.weight.mul(m) {self.final.weight.mul(m).sum(dim=2).shape}')
        # y = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)
        y = self.final(x)
        loss = self.loss_function(y, target)
        return y, loss


class DecoderOutputLayer(nn.Module):
    def __init__(self, args, Y, dicts, input_size):
        super(DecoderOutputLayer, self).__init__()

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

        loss = self.loss_function(y, target)
        return y, loss


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



class LSTM_CNN_VAE(nn.Module):

    def __init__(self, args, Y, dicts):
        super(LSTM_CNN_VAE, self).__init__()

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

        self.output_layer = DecoderOutputLayer(args, Y, dicts, self.filter_num * args.num_filter_maps)
        latent_dim = self.filter_num * args.num_filter_maps
        self.fc_mu = nn.Linear(self.filter_num * args.num_filter_maps , latent_dim)
        self.fc_var = nn.Linear(self.filter_num * args.num_filter_maps, latent_dim)
        self.emd_decoder = nn.Sequential(nn.Linear(latent_dim,latent_dim),nn.LeakyReLU(),nn.Dropout(0.2),nn.Linear(latent_dim, 100))
        #xavier_uniform(self.V_attention.weight)
        xavier_uniform(self.fc_var.weight)
        xavier_uniform(self.fc_mu.weight)
        #self.vae_fc = nn.Linear(self.filter_num * args.num_filter_maps, latent_dim)
        self.recloss = nn.MSELoss()
    def encode(self, x, target, text_inputs):
        x = self.word_rep(x, target, text_inputs)
        #print(x.shape)
        embeds = x
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
        #print(x.shape)
        # alpha = F.softmax(self.V_attention(x),dim=2)
        # x = alpha*x

        result = x#torch.mean(x,dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu,log_var,embeds

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self,x,target, text_inputs):
        y, loss = self.output_layer(x, target, text_inputs)
        #print(f' before output {x.shape} out -> {y.shape} target {target.shape} l {loss_emb}')
        return y, loss
    def forward(self, x, target, text_inputs):
        #print(x.shape,text_inputs.shape,target.shape)
        mu,logvar ,embeds= self.encode(x,target,text_inputs)
        z = self.reparameterize(mu, logvar)
        #print(z.shape)
        len = z.shape[1]*z.shape[0]
        y, loss = self.decode(z, target, text_inputs)
        reconstructedemb = self.emd_decoder(z)
        lossrec = self.recloss(reconstructedemb,embeds)
        loss_kl = get_kl_loss(mu,logvar)/len
        #print(f'loss {loss.item()} loss kl {loss_kl.item()} re {lossrec.item()}')
        #print(f' before output {x.shape} out -> {y.shape} target {target.shape} l {loss_emb}')
        return y, loss +0.01*loss_kl+lossrec

    def freeze_net(self):
        for p in self.word_rep.embed.parameters():
            p.requires_grad = False

class Residual_VAE(nn.Module):

    def __init__(self, args, Y, dicts,att=True):
        super( Residual_VAE, self).__init__()

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


        latent_dim = self.filter_num * args.num_filter_maps//2
        self.fc_mu = nn.Linear(self.filter_num * args.num_filter_maps , latent_dim)
        self.fc_var = nn.Linear(self.filter_num * args.num_filter_maps, latent_dim)
        self.output_layer = OutputLayer(args, Y, dicts, latent_dim)
        #self.emd_decoder = nn.Sequential(nn.Linear(latent_dim,latent_dim),nn.LeakyReLU(),nn.Dropout(0.2),nn.Linear(latent_dim, 100))
        #xavier_uniform(self.V_attention.weight)
        xavier_uniform(self.fc_var.weight)
        xavier_uniform(self.fc_mu.weight)
        self.att = att
        if self.att:
            self.U = nn.Linear(self.filter_num * args.num_filter_maps, latent_dim)
            xavier_uniform(self.U.weight)
        else:
            self.pool = nn.AdaptiveMaxPool1d(1)
        #self.vae_fc = nn.Linear(self.filter_num * args.num_filter_maps, latent_dim)
        #self.recloss = nn.MSELoss()

    def encode(self, x, target, text_inputs):
        x = self.word_rep(x, target, text_inputs)
        #print(x.shape)
        embeds = x
        x = x.transpose(1, 2)

        conv_result = []
        for conv in self.conv:
            tmp = x
            for idx, md in enumerate(conv):
                if idx == 0:
                    tmp = torch.tanh(md(tmp))
                else:
                    tmp = md(tmp)
            #print('conv ,',conv,'\n ',tmp.shape)
            tmp = tmp.transpose(1, 2)
            conv_result.append(tmp)
        x = torch.cat(conv_result, dim=2)
        #print(x.shape)
        # alpha = F.softmax(self.V_attention(x),dim=2)
        # x = alpha*x
        if self.att:
            alpha = F.softmax(self.U.weight.matmul(x.transpose(1, 2)), dim=2)
            #print(f'alpha {alpha.shape} x {x.shape} x.transpose {x.transpose(1, 2).shape}')
            m = alpha.matmul(x)
            result = m#torch.mean(x,dim=1) self.fc_mu.weight.t().matmul(m).sum(dim=2).add(self.fc_mu.bias).shape
            #print(self.fc_var.weight.shape,self.fc_mu.bias.shape,m.shape,(self.fc_var.weight*m).sum(dim=2).add(self.fc_mu.bias).shape)
            mu = (self.fc_mu.weight*m).sum(dim=2).add(self.fc_mu.bias)#self.fc_mu.weight.mul(m).sum(dim=2).add(self.fc_mu.bias)
            log_var = (self.fc_var.weight*m).sum(dim=2).add(self.fc_var.bias)#self.fc_var.weight.mul(m).sum(dim=2).add(self.fc_mu.bias)
        #print(f"mu {mu.shape} logvar {log_var.shape}")
        else:
            #print(x.shape)

            #
            # ,dim=1)
            x = self.pool(x.transpose(1, 2)).squeeze(-1)
            mu = self.fc_mu(x)
            log_var = self.fc_var(x)
        return mu,log_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self,x,target, text_inputs):
        y, loss = self.output_layer(x, target, text_inputs)
        #print(f' before output {x.shape} out -> {y.shape} target {target.shape} l {loss_emb}')
        return y, loss
    def forward(self, x, target, text_inputs):
        #print(x.shape,text_inputs.shape,target.shape)
        mu,logvar = self.encode(x,target,text_inputs)
        z = self.reparameterize(mu, logvar)
        #print(z.shape)
        len = z.shape[0]
        y, loss = self.decode(z, target, text_inputs)

        loss_kl = get_kl_loss(mu,logvar)/len

        return y, [loss ,loss_kl]

    def freeze_net(self):
        for p in self.word_rep.embed.parameters():
            p.requires_grad = False

class Seq2Seq_VAE(nn.Module):

    def __init__(self, args, Y, dicts,att=False):
        super(Seq2Seq_VAE, self).__init__()

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


        latent_dim = self.filter_num * args.num_filter_maps//2
        self.fc_mu = nn.Linear(self.filter_num * args.num_filter_maps , latent_dim)
        self.fc_var = nn.Linear(self.filter_num * args.num_filter_maps, latent_dim)
        self.output_layer = OutputLayer(args, Y, dicts, latent_dim)
        self.num_layers = 2
        self.hidden_size = latent_dim
        self.decoder = nn.LSTM(input_size=latent_dim,hidden_size=latent_dim,num_layers=self.num_layers,bidirectional=False,dropout=0.3)
        self.decoder_fc = nn.Sequential(nn.Linear(latent_dim,latent_dim),nn.ReLU(),nn.Dropout(0.2),nn.Linear(latent_dim,100))
        self.latent_dim = latent_dim
        xavier_uniform(self.fc_var.weight)
        xavier_uniform(self.fc_mu.weight)
        self.att = att
        if self.att:
            self.U = nn.Linear(self.filter_num * args.num_filter_maps, latent_dim)
            xavier_uniform(self.U.weight)
        else:
            self.pool = nn.AdaptiveMaxPool1d(1)
        #self.vae_fc = nn.Linear(self.filter_num * args.num_filter_maps, latent_dim)
        self.recloss = nn.MSELoss()

    def encode(self, x, target, text_inputs):
        x = self.word_rep(x, target, text_inputs)
        #print(x.shape)
        embeds = x
        x = x.transpose(1, 2)

        conv_result = []
        for conv in self.conv:
            tmp = x
            for idx, md in enumerate(conv):
                if idx == 0:
                    tmp = torch.tanh(md(tmp))
                else:
                    tmp = md(tmp)
            #print('conv ,',conv,'\n ',tmp.shape)
            tmp = tmp.transpose(1, 2)
            conv_result.append(tmp)
        x = torch.cat(conv_result, dim=2)
        #print(x.shape)
        # alpha = F.softmax(self.V_attention(x),dim=2)
        # x = alpha*x
        if self.att:
            alpha = F.softmax(self.U.weight.matmul(x.transpose(1, 2)), dim=2)
            #print(f'alpha {alpha.shape} x {x.shape} x.transpose {x.transpose(1, 2).shape}')
            m = alpha.matmul(x)
            result = m#torch.mean(x,dim=1) self.fc_mu.weight.t().matmul(m).sum(dim=2).add(self.fc_mu.bias).shape
            #print(self.fc_var.weight.shape,self.fc_mu.bias.shape,m.shape,(self.fc_var.weight*m).sum(dim=2).add(self.fc_mu.bias).shape)
            mu = (self.fc_mu.weight*m).sum(dim=2).add(self.fc_mu.bias)#self.fc_mu.weight.mul(m).sum(dim=2).add(self.fc_mu.bias)
            log_var = (self.fc_var.weight*m).sum(dim=2).add(self.fc_var.bias)#self.fc_var.weight.mul(m).sum(dim=2).add(self.fc_mu.bias)
        #print(f"mu {mu.shape} logvar {log_var.shape}")
        else:
            #print(x.shape)

            #
            # ,dim=1)
            #with torch.no_grad():
            means,vars = self.fc_mu(x),self.fc_var(x)
            x = self.pool(x.transpose(1, 2)).squeeze(-1)
            mu = self.fc_mu(x)
            log_var = self.fc_var(x)
        return mu,log_var,embeds,means,vars

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self,zetas,x,target, text_inputs,length = 2500):
        y, loss = self.output_layer(x, target, text_inputs)
        outs,j = self.decoder(zetas)

        reconstructed = self.decoder_fc(outs)


        return y, loss,reconstructed
    def forward(self, x, target, text_inputs):
        #print(x.shape,text_inputs.shape,target.shape)
        len = x.shape[-1]
        mu,logvar ,embeds,means,vars= self.encode(x,target,text_inputs)
        z = self.reparameterize(mu, logvar)
        #with torch.no_grad():
        zetas = self.reparameterize(means,vars)
        #print(z.shape)
        #len = z.shape[0]
        y, loss,reconstructed = self.decode(zetas,z, target, text_inputs,len)
        #print(reconstructed.shape,embeds.shape)
        loss_kl = get_kl_loss(mu,logvar)/len
        loss_rec = self.recloss (reconstructed,embeds)


        return y, [loss ,loss_kl+loss_rec]

    def freeze_net(self):
        for p in self.word_rep.embed.parameters():
            p.requires_grad = False

# class Seq2Seq_VAE(nn.Module):
#
#     def __init__(self, args, Y, dicts,att=True):
#         super(Seq2Seq_VAE, self).__init__()
#
#         self.word_rep = WordRep(args, Y, dicts)
#
#         self.conv = nn.ModuleList()
#         filter_sizes = args.filter_size.split(',')
#
#         self.filter_num = len(filter_sizes)
#         for filter_size in filter_sizes:
#             filter_size = int(filter_size)
#             one_channel = nn.ModuleList()
#             tmp = nn.Conv1d(self.word_rep.feature_size, self.word_rep.feature_size, kernel_size=filter_size,
#                             padding=int(floor(filter_size / 2)))
#             xavier_uniform(tmp.weight)
#             one_channel.add_module('baseconv', tmp)
#
#             conv_dimension = self.word_rep.conv_dict[args.conv_layer]
#             for idx in range(args.conv_layer):
#                 tmp = ResidualBlock(conv_dimension[idx], conv_dimension[idx + 1], filter_size, 1, True,
#                                     args.dropout)
#                 one_channel.add_module('resconv-{}'.format(idx), tmp)
#
#             self.conv.add_module('channel-{}'.format(filter_size), one_channel)
#
#
#         latent_dim = self.filter_num * args.num_filter_maps//2
#         self.fc_mu = nn.Linear(self.filter_num * args.num_filter_maps , latent_dim)
#         self.fc_var = nn.Linear(self.filter_num * args.num_filter_maps, latent_dim)
#         self.output_layer = OutputLayer(args, Y, dicts, latent_dim)
#         self.num_layers =3
#         self.hidden_size = latent_dim
#         self.decoder = nn.LSTM(input_size=latent_dim,hidden_size=latent_dim,num_layers=self.num_layers,bidirectional=False,dropout=0.2)
#         self.decoder_fc = nn.Sequential(nn.Linear(latent_dim,latent_dim),nn.LeakyReLU(),nn.Dropout(0.1),nn.Linear(latent_dim,100))
#         self.latent_dim = latent_dim
#         xavier_uniform(self.fc_var.weight)
#         xavier_uniform(self.fc_mu.weight)
#         self.att = att
#         if self.att:
#             self.U = nn.Linear(self.filter_num * args.num_filter_maps, latent_dim)
#             xavier_uniform(self.U.weight)
#         else:
#             self.pool = nn.AdaptiveMaxPool1d(1)
#         #self.vae_fc = nn.Linear(self.filter_num * args.num_filter_maps, latent_dim)
#         self.recloss = nn.MSELoss()
#
#     def encode(self, x, target, text_inputs):
#         x = self.word_rep(x, target, text_inputs)
#         #print(x.shape)
#         embeds = x
#         x = x.transpose(1, 2)
#
#         conv_result = []
#         for conv in self.conv:
#             tmp = x
#             for idx, md in enumerate(conv):
#                 if idx == 0:
#                     tmp = torch.tanh(md(tmp))
#                 else:
#                     tmp = md(tmp)
#             #print('conv ,',conv,'\n ',tmp.shape)
#             tmp = tmp.transpose(1, 2)
#             conv_result.append(tmp)
#         x = torch.cat(conv_result, dim=2)
#         #print(x.shape)
#         # alpha = F.softmax(self.V_attention(x),dim=2)
#         # x = alpha*x
#         if self.att:
#             alpha = F.softmax(self.U.weight.matmul(x.transpose(1, 2)), dim=2)
#             #print(f'alpha {alpha.shape} x {x.shape} x.transpose {x.transpose(1, 2).shape}')
#             m = alpha.matmul(x)
#             result = m#torch.mean(x,dim=1) self.fc_mu.weight.t().matmul(m).sum(dim=2).add(self.fc_mu.bias).shape
#             #print(self.fc_var.weight.shape,self.fc_mu.bias.shape,m.shape,(self.fc_var.weight*m).sum(dim=2).add(self.fc_mu.bias).shape)
#             mu = (self.fc_mu.weight*m).sum(dim=2).add(self.fc_mu.bias)#self.fc_mu.weight.mul(m).sum(dim=2).add(self.fc_mu.bias)
#             log_var = (self.fc_var.weight*m).sum(dim=2).add(self.fc_var.bias)#self.fc_var.weight.mul(m).sum(dim=2).add(self.fc_mu.bias)
#         #print(f"mu {mu.shape} logvar {log_var.shape}")
#         else:
#             #print(x.shape)
#
#             #
#             # ,dim=1)
#             x = self.pool(x.transpose(1, 2)).squeeze(-1)
#             mu = self.fc_mu(x)
#             log_var = self.fc_var(x)
#         return mu,log_var,embeds
#
#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return eps.mul(std).add_(mu)
#
#     def decode(self,x,target, text_inputs,length = 2500):
#         y, loss = self.output_layer(x, target, text_inputs)
#         out = x.unsqueeze(0)
#
#         h =  F.tanh(torch.randn(self.num_layers,out.shape[1],self.hidden_size)).cuda()
#         c = F.tanh(torch.randn(self.num_layers,out.shape[1],self.hidden_size)).cuda()
#         outputs = []
#         for i in range(length):
#             out , (h,c) = self.decoder(out,(h,c))
#
#             outputs.append(out)
#         outs = torch.cat(outputs,dim=0).permute(1,0,2)
#         reconstructed = self.decoder_fc(outs)
#
#
#         return y, loss,reconstructed
#     def forward(self, x, target, text_inputs):
#         #print(x.shape,text_inputs.shape,target.shape)
#         len = x.shape[-1]
#         mu,logvar ,embeds= self.encode(x,target,text_inputs)
#         z = self.reparameterize(mu, logvar)
#         #print(z.shape)
#         #len = z.shape[0]
#         y, loss,reconstructed = self.decode(z, target, text_inputs,len)
#         #print(reconstructed.shape,embeds.shape)
#         loss_kl = get_kl_loss(mu,logvar)/len
#         loss_rec = self.recloss (reconstructed,embeds)
#
#
#         return y, [loss ,loss_kl+loss_rec]
#
#     def freeze_net(self):
#         for p in self.word_rep.embed.parameters():
#             p.requires_grad = False




class AttSeq2Seq_VAE(nn.Module):

    def __init__(self, args, Y, dicts,att=True):
        super(AttSeq2Seq_VAE, self).__init__()

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
        latent_dim = self.filter_num * args.num_filter_maps//2
        self.num_layers =1
        self.hidden_size = latent_dim

        self.latent_dim = latent_dim

        self.fc_mu = nn.Linear(self.filter_num * args.num_filter_maps , latent_dim)
        self.fc_var = nn.Linear(self.filter_num * args.num_filter_maps, latent_dim)
        self.decoder =  AttnDecoderRNN( input_size= latent_dim,hidden_size=self.hidden_size, output_size=100)
        self.output_layer = OutputLayer(args, Y, dicts, latent_dim)

        xavier_uniform(self.fc_var.weight)
        xavier_uniform(self.fc_mu.weight)
        self.att = att
        if self.att:
            self.U = nn.Linear(self.filter_num * args.num_filter_maps, latent_dim)
            xavier_uniform(self.U.weight)
        else:
            self.pool = nn.AdaptiveMaxPool1d(1)
        #self.vae_fc = nn.Linear(self.filter_num * args.num_filter_maps, latent_dim)
        self.recloss = nn.MSELoss()

    def encode(self, x, target, text_inputs):
        x = self.word_rep(x, target, text_inputs)
        #print(x.shape)
        embeds = x
        x = x.transpose(1, 2)

        conv_result = []
        for conv in self.conv:
            tmp = x
            for idx, md in enumerate(conv):
                if idx == 0:
                    tmp = torch.tanh(md(tmp))
                else:
                    tmp = md(tmp)
            #print('conv ,',conv,'\n ',tmp.shape)
            tmp = tmp.transpose(1, 2)
            conv_result.append(tmp)
        x = torch.cat(conv_result, dim=2)
        #print(x.shape)
        # alpha = F.softmax(self.V_attention(x),dim=2)
        # x = alpha*x
        if self.att:
            alpha = F.softmax(self.U.weight.matmul(x.transpose(1, 2)), dim=2)
            #print(f'alpha {alpha.shape} x {x.shape} x.transpose {x.transpose(1, 2).shape}')
            m = alpha.matmul(x)
            result = m#torch.mean(x,dim=1) self.fc_mu.weight.t().matmul(m).sum(dim=2).add(self.fc_mu.bias).shape
            #print(self.fc_var.weight.shape,self.fc_mu.bias.shape,m.shape,(self.fc_var.weight*m).sum(dim=2).add(self.fc_mu.bias).shape)
            mu = (self.fc_mu.weight*m).sum(dim=2).add(self.fc_mu.bias)#self.fc_mu.weight.mul(m).sum(dim=2).add(self.fc_mu.bias)
            log_var = (self.fc_var.weight*m).sum(dim=2).add(self.fc_var.bias)#self.fc_var.weight.mul(m).sum(dim=2).add(self.fc_mu.bias)
        #print(f"mu {mu.shape} logvar {log_var.shape}")
        else:
            #print(x.shape)

            #
            # ,dim=1)
            x = self.pool(x.transpose(1, 2)).squeeze(-1)
            mu = self.fc_mu(x)
            log_var = self.fc_var(x)
        return mu,log_var,embeds,x

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self,x,target, text_inputs,encoder_outputs,length = 2500):
        y, loss = self.output_layer(x, target, text_inputs)
        out = x.unsqueeze(0)
        B = x.shape[0]
        h =  F.tanh(torch.randn(self.num_layers,out.shape[1],self.hidden_size)).cuda()
        c = F.tanh(torch.randn(self.num_layers,out.shape[1],self.hidden_size)).cuda()
        outputs = []
        dec_input = x
        last_context = torch.zeros(B,self.hidden_size).cuda()
        last_hidden = torch.zeros(1,B,self.hidden_size).cuda()
        encoder_outputs = encoder_outputs.permute(1,0,2)
        for i in range(length):
            out, last_context, last_hidden, attn_weights= self.decoder(dec_input, last_context, last_hidden, encoder_outputs)

            outputs.append(out)
        outs = torch.cat(outputs,dim=0).permute(1,0,2)
        reconstructed = outs


        return y, loss,reconstructed
    def forward(self, x, target, text_inputs):
        #print(x.shape,text_inputs.shape,target.shape)
        len = x.shape[-1]
        mu,log_var,embeds,encoder_outputs= self.encode(x,target,text_inputs)
        z = self.reparameterize(mu, log_var)
        #print(z.shape)
        #len = z.shape[0]
        y, loss,reconstructed = self.decode(z, target, text_inputs,encoder_outputs,len)
        #print(reconstructed.shape,embeds.shape)
        loss_kl = get_kl_loss(mu,log_var)/len
        loss_rec = self.recloss (reconstructed,embeds)


        return y, [loss ,loss_kl+loss_rec]

    def freeze_net(self):
        for p in self.word_rep.embed.parameters():
            p.requires_grad = False

class Dense_VAE(nn.Module):
    def __init__(self, args, Y, dicts,K=7):
        super(Dense_VAE, self).__init__()
        self.word_rep = WordRep(args, Y, dicts)
        filters = [100]
        dc =200
        for i in range(2,K+1):
            filters += [dc]
            print(filters,sum(filters[:-1]))
            self.add_module(f"block{i-2}",DenseBlock(sum(filters[:-1]),filters[i-1],3))
        self.output_layer = OutputLayer(args, Y, dicts,dc)
        latent_dim = dc
        self.fc_mu = nn.Linear(dc , latent_dim)
        self.fc_var = nn.Linear(dc, latent_dim)
    def encode(self,x, target, text_inputs):
        x = self.word_rep(x, target, text_inputs)
        x = x.transpose(1, 2)
        x1 = self.block0(x)
        x2 = self.block1(torch.cat((x1,x),dim=1))

        x3 = self.block2(torch.cat((x2,x1,x),dim=1))
        x4 = self.block3(torch.cat((x3,x2,x1,x),dim=1))
        x5 = self.block4(torch.cat((x4,x3,x2,x1,x),dim=1))
        x6 = self.block5(torch.cat((x5,x4, x3, x2, x1, x), dim=1))

        #x6 = x6.transpose(1, 2)
        result = torch.mean(x6,dim=-1)
        var = self.fc_var(result)
        mu = self.fc_mu(result)
        return mu,var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)


    def decode(self,x,target, text_inputs):
        y, loss = self.output_layer(x, target, text_inputs)
        #print(f' before output {x.shape} out -> {y.shape} target {target.shape} l {loss_emb}')
        return y, loss
    def forward(self, x, target, text_inputs):


        mu,logvar = self.encode(x,target,text_inputs)
        z = self.reparameterize(mu, logvar)
        out, loss = self.decode(z,target,text_inputs)
        loss_kl = get_kl_loss(mu, logvar)
        return out,loss+0.001*loss_kl

class DenseNet_VAE(nn.Module):
    def __init__(self,args, Y, dicts):
        super(DenseNet_VAE, self).__init__()
        self.word_rep = WordRep(args, Y, dicts)
        #self.tcn = TemporalConvNet(100,[150,150,150,100])
        self.cnn = densenet100()
        nf = self.cnn.final_num_features




        latent_dim = nf//2
        self.fc_mu = nn.Linear(nf , latent_dim)
        self.fc_var = nn.Linear(nf, latent_dim)
        self.output_layer = nn.Linear(latent_dim,Y)
        self.loss = nn.BCEWithLogitsLoss()
    def encode(self,x, target, text_inputs):
        x = self.word_rep(x, target, text_inputs)
        #print(x.shape)
        x = x.transpose(1, 2)
        #x = self.tcn(x)
        out = self.cnn(x)

        #x6 = x6.transpose(1, 2)
        result = torch.mean(out,dim=-1)
        var = self.fc_var(result)
        mu = self.fc_mu(result)
        return mu,var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)


    def decode(self,x,target, text_inputs):
        #print(x.shape)
        y= self.output_layer(x)
        loss = self.loss(y, target)
        #print(f' before output {x.shape} out -> {y.shape} target {target.shape} l {loss_emb}')
        return y, loss
    def forward(self, x, target, text_inputs):


        mu,logvar = self.encode(x,target,text_inputs)
        z = self.reparameterize(mu, logvar)
        out, loss = self.decode(z,target,text_inputs)
        loss_kl = get_kl_loss(mu, logvar)/z.shape[0]
        return out,[loss,loss_kl]


class MultiAttVAE(nn.Module):
    def __init__(self, args, Y, dicts,K=7,attn = 'bandanau'):
        super(MultiAttVAE, self).__init__()
        self.word_rep = WordRep(args, Y, dicts)

        filters = [100]
        dc =200
        self.attn = attn
        self.latent_dim = dc //2
        if attn == 'bandanau':
            for i in range(2, K + 1):
                filters += [dc]
                print(filters, sum(filters[:-1]))
                self.add_module(f"block{i - 2}", DenseBlock(sum(filters[:-1]), filters[i - 1], 3))
                self.add_module(f"U{i - 2}",nn.Linear(dc,self.latent_dim))


        else:
            for i in range(2,K+1):
                filters += [dc]
                print(filters,sum(filters[:-1]))
                self.add_module(f"block{i-2}",DenseBlock(sum(filters[:-1]),filters[i-1],3))
                self.add_module(f"U{i-2}",nn.Linear(dc,Y))

        self.fc_mu = nn.Linear(self.latent_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.latent_dim, self.latent_dim)
        self.output_layer = nn.Linear( self.latent_dim,Y)
        self.loss_function = nn.BCEWithLogitsLoss()
    def encode(self,x, target, text_inputs):
        x = self.word_rep(x, target, text_inputs)
        x = x.transpose(1, 2)
        x1 = self.block0(x)
        x2 = self.block1(torch.cat((x1,x),dim=1))

        x3 = self.block2(torch.cat((x2,x1,x),dim=1))
        x4 = self.block3(torch.cat((x3,x2,x1,x),dim=1))
        x5 = self.block4(torch.cat((x4,x3,x2,x1,x),dim=1))
        x6 = self.block5(torch.cat((x5,x4, x3, x2, x1, x), dim=1))

        alpha1 = F.softmax(self.U0.weight.matmul(x1), dim=2).matmul(x1.transpose(1, 2))
        # print(alpha1.shape,x1.shape)
        alpha2 = (F.softmax(self.U1.weight.matmul(x2), dim=2)).matmul(x2.transpose(1, 2))
        alpha3 = (F.softmax(self.U2.weight.matmul(x3), dim=2)).matmul(x3.transpose(1, 2))
        alpha4 = (F.softmax(self.U3.weight.matmul(x4), dim=2)).matmul(x4.transpose(1, 2))
        alpha5 = (F.softmax(self.U4.weight.matmul(x5), dim=2)).matmul(x5.transpose(1, 2))
        alpha6 = (F.softmax(self.U5.weight.matmul(x6), dim=2)).matmul(x6.transpose(1, 2))
        print(f"alpha {alpha1.shape} {self.fc_mu.weight.mul(alpha1).shape}")
        y = self.fc_mu.weight.mul(alpha1).sum(dim=1).add(self.fc_mu.bias)
        print(y.shape)
        var = y[:,:self.latent_dim//2]
        mu = y[:,self.latent_dim//2:]
        return mu,var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    def decode(self,x,target, text_inputs):
        #print(x.shape)
        y = self.output_layer(x)#, target, text_inputs)
        #print(f' before output {x.shape} out -> {y.shape} target {target.shape} l {loss_emb}')
        return y, self.loss_function(y,target)

    def forward(self, x, target, text_inputs):

        mu, logvar = self.encode(x, target, text_inputs)
        z = self.reparameterize(mu, logvar)
        out, loss = self.decode(z, target, text_inputs)
        loss_kl = get_kl_loss(mu, logvar)/z.shape[0]
        #print(f"loss {loss.item()} kl {loss_kl.item()}")
        return out, [loss , loss_kl]



class MultiScaleAttVAE(nn.Module):
    def __init__(self, args, Y, dicts,K=7,attn = 'bandanau'):
        super(MultiScaleAttVAE, self).__init__()
        self.word_rep = WordRep(args, Y, dicts)

        filters = [100]
        dc = 100
        self.attn = attn
        if attn == 'bandanau':
            for i in range(2, K + 1):
                filters += [dc]
                print(filters, sum(filters[:-1]))
                self.add_module(f"block{i - 2}", DenseBlock(sum(filters[:-1]), filters[i - 1], 3))
            self.multi_att = MultiScaleAttention(K-1)
            self.U = nn.Linear(dc,dc)
        else:
            for i in range(2,K+1):
                filters += [dc]
                print(filters,sum(filters[:-1]))
                self.add_module(f"block{i-2}",DenseBlock(sum(filters[:-1]),filters[i-1],3))
                self.add_module(f"U{i-2}",nn.Linear(dc,Y))

        self.latent_dim = dc
        self.fc_latent = nn.Linear(dc,self.latent_dim)
        self.output_layer = nn.Linear( self.latent_dim//2,Y)
        self.loss_function = nn.BCEWithLogitsLoss()
    def encode(self,x, target, text_inputs):
        x = self.word_rep(x, target, text_inputs)
        x = x.transpose(1, 2)
        x1 = self.block0(x)
        x2 = self.block1(torch.cat((x1,x),dim=1))

        x3 = self.block2(torch.cat((x2,x1,x),dim=1))
        x4 = self.block3(torch.cat((x3,x2,x1,x),dim=1))
        x5 = self.block4(torch.cat((x4,x3,x2,x1,x),dim=1))
        x6 = self.block5(torch.cat((x5,x4, x3, x2, x1, x), dim=1))

        #x6 = x6.transpose(1, 2)
        concat = torch.stack((x1, x2, x3, x4, x5, x6), dim=-1)
        x_attn = self.multi_att(concat)  # .transpose(1,2)

        alpha = F.softmax(self.U.weight.matmul(x_attn.transpose(1, 2)), dim=2).matmul(x_attn)
        #print(f"alpha {alpha.shape}")
        y = self.fc_latent.weight.mul(alpha).sum(dim=2).add(self.fc_latent.bias)
        var = y[:,:self.latent_dim//2]
        mu = y[:,self.latent_dim//2:]
        return mu,var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    def decode(self,x,target, text_inputs):
        #print(x.shape)
        y = self.output_layer(x)#, target, text_inputs)
        #print(f' before output {x.shape} out -> {y.shape} target {target.shape} l {loss_emb}')
        return y, self.loss_function(y,target)

    def forward(self, x, target, text_inputs):

        mu, logvar = self.encode(x, target, text_inputs)
        z = self.reparameterize(mu, logvar)
        out, loss = self.decode(z, target, text_inputs)
        loss_kl = get_kl_loss(mu, logvar)/z.shape[0]
        #print(f"loss {loss.item()} kl {loss_kl.item()}")
        return out, [loss , loss_kl]



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

def pick_model(args, dicts):
    Y = len(dicts['ind2c'])

    if args.model == 'Dense_VAE':
        model = Dense_VAE(args, Y, dicts)
    elif args.model == 'Residual_VAE':
        model = Residual_VAE(args,Y,dicts)
    elif args.model =="MultiScaleAttVAE":
        model = MultiScaleAttVAE(args,Y,dicts)
    elif args.model== 'DenseNet_VAE':
        model = DenseNet_VAE(args,Y,dicts)
    elif args.model == 'MultiAttVAE':
        model = MultiAttVAE(args,Y,dicts)
    elif args.model == 'Seq2Seq_VAE':
        model = Seq2Seq_VAE(args,Y,dicts)
    elif args.model == 'AttSeq2Seq_VAE':
        model = AttSeq2Seq_VAE(args,Y,dicts)
    else:
        raise RuntimeError("wrong model name")

    if args.test_model:
        sd = torch.load(args.test_model)
        model.load_state_dict(sd)
    if args.gpu >= 0:
        model.cuda(args.gpu)
    return model
