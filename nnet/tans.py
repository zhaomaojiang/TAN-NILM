#!/usr/bin/env python

import torch as th
import torch.nn as nn
import torch.nn.functional as F


from .norm import ChannelwiseLayerNorm, GlobalLayerNorm
from .cnns import Conv1D, ConvTrans1D, TCNBlock, TCNBlock_Spk, ResBlock, TCNBlock_Spk1, TCNBlock_Spk2
from torch.nn.modules.container import ModuleList
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTM
from torch.nn.modules.module import Module
from torch.nn.modules.normalization import LayerNorm


class TransformerEncoderLayer(Module):
    """TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = LSTM(d_model, d_model * 2, 1, bidirectional=True)
#         self.linear1 = Linear(d_model, d_model * 2 * 2)
        self.linear2 = Linear(d_model * 2 * 2, d_model)
        self.dropout = nn.Dropout(dropout)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        self.activation = nn.PReLU()

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        ## type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        self.linear1.flatten_parameters()  # 将LSTM层的权重参数展开以提高效率
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src)[0])))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TAN_Plus(nn.Module):
    def __init__(self,
                 L1=20,
                 L2=80,
                 L3=160,
                 N=256,
                 B=8,
                 O=256,
                 P=512,
                 Q=3,  # kernel size
                 num_spks=9,
                 spk_embed_dim=256,
                 Head=4,
                 causal=False):
        super(TAN_Plus, self).__init__()
        # n x S => n x N x T, S = 4s*8000 = 32000
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3
        self.encoder_1d_short = Conv1D(1, N, L1, stride=L1 // 2, padding=0)
        self.encoder_1d_middle = Conv1D(1, N, L2, stride=L1 // 2, padding=0)
        self.encoder_1d_long = Conv1D(1, N, L3, stride=L1 // 2, padding=0)
        # before repeat blocks, always cLN
        self.ln = ChannelwiseLayerNorm(N)
        # n x N x T => n x O x T
        self.proj = Conv1D(N, O, 1)
        self.conv_block_1 = TCNBlock_Spk1(in_channels=O, spk_embed_dim=spk_embed_dim, conv_channels=P, kernel_size=Q,
                                          dilation=1, causal=causal)
        self.conv_block_1_other = self._build_stacks(num_blocks=B, in_channels=O, conv_channels=P, kernel_size=Q, causal=causal)
        self.conv_block_2 = TCNBlock_Spk1(in_channels=O, spk_embed_dim=spk_embed_dim, conv_channels=P, kernel_size=Q,
                                          dilation=1, causal=causal)
        self.conv_block_2_other = self._build_stacks(num_blocks=B, in_channels=O, conv_channels=P, kernel_size=Q, causal=causal)
        self.conv_block_3 = TCNBlock_Spk1(in_channels=O, spk_embed_dim=spk_embed_dim, conv_channels=P, kernel_size=Q,
                                          dilation=1, causal=causal)
        self.conv_block_3_other = self._build_stacks(num_blocks=B, in_channels=O, conv_channels=P, kernel_size=Q, causal=causal)
        self.conv_block_4 = TCNBlock_Spk1(in_channels=O, spk_embed_dim=spk_embed_dim, conv_channels=P, kernel_size=Q,
                                          dilation=1, causal=causal)
        self.conv_block_4_other = self._build_stacks(num_blocks=B, in_channels=O, conv_channels=P, kernel_size=Q, causal=causal)
        # n x O x T => n x N x T
        self.mask1 = Conv1D(O, N, 1)
        self.mask2 = Conv1D(O, N, 1)
        self.mask3 = Conv1D(O, N, 1)
        # using ConvTrans1D: n x N x T => n x 1 x To
        # To = (T - 1) * L // 2 + L
        self.decoder_1d_short = ConvTrans1D(N, 1, kernel_size=L1, stride=L1 // 2, bias=True)
        self.decoder_1d_middle = ConvTrans1D(N, 1, kernel_size=L2, stride=L1 // 2, bias=True)
        self.decoder_1d_long = ConvTrans1D(N, 1, kernel_size=L3, stride=L1 // 2, bias=True)
        self.num_apps = num_spks

        self.spk_encoder = nn.Sequential(
            ChannelwiseLayerNorm(N),
            Conv1D(N, O, 1),
            ResBlock(O, O),
            ResBlock(O, P),
            ResBlock(P, P),
            Conv1D(P, spk_embed_dim, 1),
        )

        self.pred_linear = nn.Linear(spk_embed_dim, num_spks)
        self.att = TransformerEncoderLayer(d_model=N, nhead=Head, dim_feedforward=P, dropout=0.9)
        # self.att1 = TransformerEncoderLayer(d_model=N, nhead=Head, dim_feedforward=P, dropout=0.5)
        # self.att2 = TransformerEncoderLayer(d_model=N, nhead=Head, dim_feedforward=P, dropout=0)
        # self.att3 = TransformerEncoderLayer(d_model=N, nhead=Head, dim_feedforward=P, dropout=0)
        # self.linear = nn.Linear(430, 1024)
        # self.linear1 = nn.Linear(1024, 30)

    def _build_stacks(self, num_blocks, **block_kwargs):
        """
        Stack B numbers of TCN block, the first TCN block takes the speaker embedding
        """
        blocks = [
            TCNBlock(**block_kwargs, dilation=(2**b))
            for b in range(1,num_blocks)
        ]
        return nn.Sequential(*blocks)

    def forward(self, x, aux, aux_len):
        if x.dim() >= 3:
            raise RuntimeError(
                "{} accept 1/2D tensor as input, but got {:d}".format(
                    self.__name__, x.dim()))
        # when inference, only one utt
        if x.dim() == 1:
            x = th.unsqueeze(x, 0)

        # n x 1 x S => n x N x T
        w1 = F.relu(self.encoder_1d_short(x))
        T = w1.shape[-1]
        xlen1 = x.shape[-1]

        # n x 3N x T
        y = self.ln(w1)
        # n x O x T
        y = self.proj(y)

        # speaker encoder (share params from speech encoder)
        aux_w1 = F.relu(self.encoder_1d_short(aux))
        aux_T_shape = aux_w1.shape[-1]
        aux_len1 = aux.shape[-1]

        aux = self.spk_encoder(aux_w1)
        aux_T = (aux_len - self.L1) // (self.L1 // 2) + 1
        # aux_T = th.div((aux_len - self.L1), (self.L1 // 2), rounding_mode='trunc') - 1
        # aux_T = ((aux_T // 3) // 3) // 3
        aux = th.sum(aux, -1)/aux_T.view(-1,1).float()

        y = self.conv_block_1(y, aux)
        y = self.conv_block_1_other(y)
        y = self.conv_block_2(y, aux)
        y = self.conv_block_2_other(y)
        y = self.conv_block_3(y, aux)
        y = self.conv_block_3_other(y)
        y = self.conv_block_4(y, aux)
        y = self.conv_block_4_other(y)
        y = y.permute(0, 2, 1).contiguous()
        y = self.att(y)
        # y = self.att1(y)
        # y = self.att2(y)
        # y = self.att3(y)
        # y = self.linear(y)
        # y = self.linear1(y)
        y = y.permute(0, 2, 1).contiguous()

        # n x N x T
        m1 = F.relu(self.mask1(y))
        S1 = w1 * m1
        # S1 = self.linear(S1)
        # S1 = self.linear1(S1)

        return self.decoder_1d_short(S1), self.pred_linear(aux)

    def predce(self, aux, aux_len):
        aux_w1 = F.relu(self.encoder_1d_short(aux))
        aux = self.spk_encoder(aux_w1)
        aux_T = (aux_len - self.L1) // (self.L1 // 2) + 1
        # aux_T = th.div((aux_len - self.L1), (self.L1 // 2), rounding_mode='trunc')-1
        # aux_T = ((aux_T // 3) // 3) // 3
        aux = th.sum(aux, -1) / aux_T.view(-1, 1).float()
        return self.pred_linear(aux)


def compute_loss(est, ref, onehot):
    """计算MSE loss以及CrossEntropy loss"""
    lossfunc = th.nn.MSELoss()
    ce = th.nn.CrossEntropyLoss()
    snr1 = lossfunc(est[0], ref)
    snr_loss = th.sum(snr1)
    ce_loss = ce(est[1], onehot)
    return snr_loss, ce_loss


def compute_ce_loss(est, onehot):
    """计算CrossEntropy loss"""
    ce = th.nn.CrossEntropyLoss()
    ce_loss = ce(est, onehot)
    return ce_loss


def compute_mae(est ,ref, onehot):
    lossfunc = th.nn.L1Loss(reduction='mean')
    ce = th.nn.CrossEntropyLoss()
    snr1 = lossfunc(est[0], ref)
    snr_loss = th.sum(snr1)
    ce_loss = ce(est[1], onehot)
    return snr_loss, ce_loss


def sisdr(est ,ref, onehot):
    x_zm = est[0] - th.mean(est[0], dim=-1, keepdim=True)
    s_zm = ref - th.mean(ref, dim=-1, keepdim=True)
    eps = 1e-8
    t = th.sum(x_zm * s_zm, dim=-1, keepdim=True) * s_zm / (th.norm(s_zm, dim=-1, keepdim=True) ** 2 + eps)
    snr_loss = -th.sum(20 * th.log10(eps + th.norm(t, dim=-1, keepdim=True) / (th.norm(x_zm - t, dim=-1, keepdim=True) + eps)))
    ce = th.nn.CrossEntropyLoss()
    ce_loss = ce(est[1], onehot)
    return snr_loss, ce_loss


def compute_loss_val(est, ref, onehot):
    """ 计算L1loss以及CEloss"""
    lossfunc = th.nn.L1Loss(reduction='mean')  # MAE 平均绝对误差
    ce = th.nn.CrossEntropyLoss()
    snr1 = lossfunc(est[0], ref)
    snr_loss = th.sum(snr1)
    ce_loss = ce(est[1], onehot)
    return snr_loss, ce_loss


if __name__ == '__main__':
    input = th.rand(64, 512)
    aux = th.rand(64,256)
    si = th.ones(32)*256
    s2 = th.ones(32)*1024
    si = th.cat([si, s2])
    si = list(si)
    si = th.tensor(si)
    model = TAN_Plus(L1=3, L2=21, L3=41, N=256, B=3, O=256, P=512, Q=3, num_spks=9, spk_embed_dim=256, causal=False)
    # print(model)
    out = model(input, aux, si)
    # print(out.shape)

    k = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('# of parameters:', k)