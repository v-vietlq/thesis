import os
from torch import nn
import torch
# from fastai2.layers import trunc_normal_
from utils.utils import trunc_normal_
import pickle
import copy


class TransformerEncoderLayerWithWeight(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super(TransformerEncoderLayerWithWeight,
              self).__init__(*args, **kwargs)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, attn_weight = self.self_attn(src, src, src, attn_mask=src_mask,
                                           key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_weight


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerEncoderWithWeight(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoderWithWeight, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src):
        output = src

        for mod in self.layers:
            output = mod(output)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TAtentionAggregate(nn.Module):
    def __init__(self, clip_length=None, enc_layer=None, embed_dim=2048, n_layers=6, args=None):
        super(TAtentionAggregate, self).__init__()
        self.clip_length = clip_length
        drop_rate = 0.
        self.args = args
        self.transformer_enc = TransformerEncoderWithWeight(enc_layer, num_layers=n_layers,
                                                            norm=nn.LayerNorm(
                                                                embed_dim))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, clip_length + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        with torch.no_grad():
            trunc_normal_(self.pos_embed, std=.02)
            trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        self.clip_length = x.shape[0] if self.args.infer == True else self.clip_length
        nvids = x.shape[0] // self.clip_length
        x = x.view((nvids, self.clip_length, -1))
        pre_aggregate = torch.clone(x)
        cls_tokens = self.cls_token.expand(nvids, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.args.transformers_pos:
            x = x + self.pos_embed
        # x = self.pos_drop(x)

        x.transpose_(1, 0)
        o = self.transformer_enc(x)
        o.transpose_(1, 0)

        return o[:, 0]
