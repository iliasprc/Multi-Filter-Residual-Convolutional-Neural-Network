import torch
import torch.nn as nn
from torch import Tensor
from typing import Any, List, Tuple
import torch.nn.functional as F
from collections import OrderedDict
# from models.tcn import TemporalConvNet,TemporalCnn
import torch.nn.init
from elmo.elmo import Elmo
import json
# from models.attn import Attn
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

class _DenseLayer(nn.Module):
    def __init__(
        self,
        num_input_features: int,
        growth_rate: int,
        bn_size: int,
        drop_rate: float,
        memory_efficient: bool = False
    ) -> None:
        super(_DenseLayer, self).__init__()
        self.norm1: nn.BatchNorm1d
        self.add_module('norm1', nn.BatchNorm1d(num_input_features))
        self.relu1: nn.ReLU
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.conv1: nn.Conv1d
        self.add_module('conv1', nn.Conv1d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False))
        self.norm2: nn.BatchNorm1d
        self.add_module('norm2', nn.BatchNorm1d(bn_size * growth_rate))
        self.relu2: nn.ReLU
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.conv2: nn.Conv1d
        self.add_module('conv2', nn.Conv1d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False))
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs: List[Tensor]) -> Tensor:
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output






    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input: Tensor) -> Tensor:  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input


        bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features


class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        drop_rate: float,
        memory_efficient: bool = False
    ) -> None:
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features: Tensor) -> Tensor:
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm1d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv1d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        #self.add_module('pool', nn.MaxPool1d(kernel_size=2, stride=2))



class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """

    def __init__(
        self,
        growth_rate: int = 32,
        block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
        num_init_features: int = 64,
        num_classes: int = 50,
        bn_size: int = 4,
        drop_rate: float = 0,

        memory_efficient: bool = False
    ) -> None:

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv1d(100, num_init_features, kernel_size=3, stride=1,
                                padding=1, bias=False)),
            ('norm0', nn.BatchNorm1d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            #('pool0', nn.MaxPool1d(kernel_size=2, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm1d(num_features))
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(1)

        # Linear layer
        self.final_num_features = num_features
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        features = self.features(x)
        out = F.relu(features, inplace=True)

        # out = self.avg_pool(out)
        #
        #
        # out = out.squeeze(-1)
        #
        # out = self.classifier(out)
        return out


def _densenet(
    arch: str,
    growth_rate: int,
    block_config: Tuple[int, int, int, int],
    num_init_features: int,
    num_classes:int,
    pretrained: bool,
    progress: bool,

    **kwargs: Any
) -> DenseNet:
    model = DenseNet(growth_rate, block_config, num_init_features,num_classes, **kwargs)

    return model


def densenet121(pretrained: bool = False, progress: bool = True,num_classes=50, **kwargs: Any) -> DenseNet:
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """
    return _densenet('densenet121', 32, (6, 12, 24, 16), 64,num_classes, pretrained, progress,
                     **kwargs)
def densenet100(pretrained: bool = False, progress: bool = True,num_classes=50, **kwargs: Any) -> DenseNet:
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """
    return _densenet('densenet100', 32, (6, 6,6,6), 128,num_classes, pretrained, progress,
                     **kwargs)

class OutputLayer(nn.Module):
    def __init__(self, args, Y, dicts, input_size):
        super(OutputLayer, self).__init__()

        self.U = nn.Linear(input_size, Y)
        torch.nn.init.xavier_uniform(self.U.weight)


        self.final = nn.Linear(input_size, Y)
        torch.nn.init.xavier_uniform(self.final.weight)

        self.loss_function = nn.BCEWithLogitsLoss()



    def forward(self, x, target, text_inputs):
        print(f'Inside out layer {x.shape} x.transpose {x.transpose(1, 2).shape}')
        alpha = F.softmax(self.U.weight.matmul(x.transpose(1, 2)), dim=2)
        #print(f'alpha {alpha.shape} x {x.shape} x.transpose {x.transpose(1, 2).shape}')
        m = alpha.matmul(x)
        print(f'alpha {alpha.shape} x {x.shape} x.transpose {x.transpose(1, 2).shape} m {m.shape}')
        y = self.final.weight.mul(m).sum(dim=2).add(self.final.bias)
        #y = self.final(m.mean(dim=1))

        loss = self.loss_function(y, target)
        return y, loss
class TCN(nn.Module):
    def __init__(self, args, Y, dicts):
        super(TCN,self).__init__()
        self.word_rep = WordRep(args, Y, dicts)
        #self.tcn = TemporalConvNet(100,11*[150])
        self.tcn = TemporalCnn(100, [100,150,150,200,200,250,250,300,300])
        self.output_layer = OutputLayer(args, Y, dicts,300)

    def forward(self, x, target, text_inputs):
        # print(x.shape,text_inputs.shape,target.shape)
        x = self.word_rep(x, target, text_inputs)
        x = x.transpose(1, 2)
        x = self.tcn(x)
        #print(x.shape)
        x = x.transpose(1, 2)
        y, loss = self.output_layer(x, target, text_inputs)

        return y, loss#+loss_emb


class Dense_CNN(nn.Module):
    def __init__(self,args, Y, dicts):
        super(Dense_CNN, self).__init__()
        self.word_rep = WordRep(args, Y, dicts)
        #self.tcn = TemporalConvNet(100,[150,150,150,100])
        self.cnn = densenet100()
        nf = self.cnn.final_num_features
        self.att = Attn('bmm',nf)
        #self.output_layer = OutputLayer(args, Y, dicts, nf)
        self.output_layer = nn.Linear(nf,Y)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x, target, text_inputs):
        # print(x.shape,text_inputs.shape,target.shape)
        x = self.word_rep(x, target, text_inputs)
        #print(x.shape)
        x = x.transpose(1, 2)
        #x = self.tcn(x)
        out = self.cnn(x)
        x = self.att(out.permute(0,2,1))
        #print(out.shape)
        #x = out.transpose(1, 2)
        y = self.output_layer(x)
        loss = self.loss(y,target)
        return y,loss
