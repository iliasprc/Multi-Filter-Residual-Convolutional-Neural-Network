import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class Attn(nn.Module):
    def __init__(self,method,  hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size , hidden_size,bias=False)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self,  encoder_outputs, src_len=None):
        '''
      :param encoder_outputs:
            encoder outputs from Encoder, in shape (B,T,H)
        :param src_len:
            used for masking. NoneType or tensor in shape (B) indicating sequence length
        :return
            attention energies in shape (B,T)
        '''
        max_len = encoder_outputs.size(1)
        this_batch_size = encoder_outputs.size(2)


        attn_energies = self.score( encoder_outputs)  # compute attention score
        #print(attn_energies.shape)

        if self.method == 'bmm':
            attn_w = torch.softmax(attn_energies, dim=-1).unsqueeze(1)
            out = attn_w.bmm(encoder_outputs).squeeze(1)
            return  out # normalize with softmax
        else:
            attn_w = torch.softmax(attn_energies, dim=-1).unsqueeze(-1)
            #print(attn_w.shape,encoder_outputs.shape)
            return attn_w * encoder_outputs


    def score(self,  encoder_outputs):
        energy = torch.tanh(self.attn( encoder_outputs))  # [B*T*H]
        energy = energy.transpose(2, 1)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.data.shape[0], 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]
#
# m = Attn('',100)
# inp = torch.randn(8,1000,100)
# out = m(inp)
