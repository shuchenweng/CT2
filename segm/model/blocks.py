"""
Adapted from 2020 Ross Wightman
https://github.com/rwightman/pytorch-image-models
"""

import torch
import torch.nn as nn
from einops import rearrange
from pathlib import Path
import time
import torch.nn.functional as F
import numpy as np
from timm.models.layers import DropPath



class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout, out_dim=None):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        if out_dim is None:
            out_dim = dim
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):         # for vit attention
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.attn = None

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x, mask=None):
        B, N, C = x.shape       # x: [B, 16*16+313, C]
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.heads, C // self.heads)
            .permute(2, 0, 3, 1, 4)     # [3, B, self.heads, N, C//heads]
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale       # [B, heads, 16*16+313, 16*16+313]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, mask=None, return_attention=False):
        y, attn = self.attn(self.norm1(x), mask)
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Decoder_Attention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.attn = None
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x, mask=None, without_colorattn=False):
        if not without_colorattn:
            B, N, C = x.shape       # x: [B, 16*16+313, C]
            qkv = (
                self.qkv(x)
                .reshape(B, N, 3, self.heads, C // self.heads)
                .permute(2, 0, 3, 1, 4)     # [3, B, self.heads, N, C//heads]
            )
            q, k, v = (
                qkv[0],
                qkv[1],
                qkv[2],
            )
            # q,k,v: [B, heads, 16*16+313, C//heads] , heads = 3

            attn = (q @ k.transpose(-2, -1)) * self.scale       # [B, heads, 16*16+313, 16*16+313]

            if mask is not None:        # add_mask == True
                expand_mask = mask.unsqueeze(1).repeat(1, self.heads, 1, 1)     # [B, heads, 16*16+313, 16*16+313]
                attn = attn.masked_fill(expand_mask == 0, -float('inf'))

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
        else:
            assert without_colorattn is True
            B, N, C = x.shape
            p_num, n_cls = 16*16, 313
            patch_tokens, color_tokens = x[:, :-n_cls, :], x[:, p_num:, :]
            qkv = (self.qkv(patch_tokens).reshape(B, p_num, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4))
            q, k, v = (qkv[0], qkv[1], qkv[2])
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            patches = (attn @ v).transpose(1, 2).reshape(B, p_num, C)
            patches = self.proj(patches)
            patches = self.proj_drop(patches)
            x = torch.cat((patches, color_tokens), dim=1)       # [B, N, C]

        return x, attn


class Decoder_Block(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Decoder_Attention(dim, heads, dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, mask=None, return_attention=False, without_colorattn=False):
        y, attn = self.attn(self.norm1(x), mask, without_colorattn=without_colorattn)
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


################################## For multi-scaled Decoder ###############################
class PixelNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=2, keepdim=True) + 1e-8)


class Multidecoder_Attention(nn.Module):
    def __init__(self, dim, heads, dropout, num_windows):
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.attn = None
        self.num_windows = num_windows

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, patch_tokens, color_tokens, mask):
        B, N, C = patch_tokens.shape
        n_cls = color_tokens.size(1)
        batchsize = B//self.num_windows
        patch_tokens = patch_tokens.view(batchsize, self.num_windows, N, C)
        new_patch_tokens = torch.zeros_like(patch_tokens)
        mask = mask.view(batchsize, self.num_windows, N+n_cls, N+n_cls)
        for win in range(self.num_windows):
            x = torch.cat((patch_tokens[:, win], color_tokens), dim=1)      #[batchsize, N+313, C]
            qkv = (self.qkv(x).reshape(batchsize, N+n_cls, 3, self.heads, C//self.heads).permute(2, 0, 3, 1, 4))
            q, k, v = (qkv[0], qkv[1], qkv[2])      # [batchsize, heads, 16*16+313, C//heads]
            attn = (q @ k.transpose(-2, -1)) * self.scale       # [B, heads, 16*16+313, 16*16+313]
            expand_mask = mask[:, win].unsqueeze(1).repeat(1, self.heads, 1, 1)     # [batchsize, heads, N+313, N+313]
            attn = attn.masked_fill(expand_mask == 0, -float('inf'))
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(batchsize, N+n_cls, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            color_tokens = x[:, -n_cls:]        # [bs, 313, C]
            new_patch_tokens[:, win] = x[:, :N]
        #################################################################
        return new_patch_tokens.view(B, N, C), color_tokens


class CustomNorm(nn.Module):
    def __init__(self, norm_layer, dim):
        super(CustomNorm, self).__init__()
        self.norm_type = norm_layer
        if norm_layer == 'ln':
            self.norm = nn.LayerNorm(dim)
        elif norm_layer == 'bn':
            self.norm = nn.BatchNorm1d(dim)
        elif norm_layer == 'in':
            self.norm = nn.InstanceNorm1d(dim)
        elif norm_layer == 'pn':
            self.norm = PixelNorm(dim)

    def forward(self, x):
        if self.norm_type == 'bn' or self.norm_type == 'in':
            x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
            return x
        elif self.norm_type == 'none':
            return x
        else:
            return self.norm(x)


def window_partition(x, window_size):
    # only for patch tokens.
    B, H, W, C = x.shape
    x = x.view(B, H//window_size, window_size, W//window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    # [B * H//w_size * W//w_size, W_size, W_size, C]
    return windows


def window_reverse(windows, window_size, H, W):
    # windows: (num_windows*B, window_size, window_size, C)->x: (B, H, W, C)
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Multiscale_Block(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout, drop_path, window_size, n_cls, num_windows, norm_layer='pn'):
        super(Multiscale_Block, self).__init__()
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Multidecoder_Attention(dim, heads, dropout, num_windows)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.num_windows = num_windows
        self.n_cls = n_cls          # 313

    def process_mask(self, mask, N_patch):
        # mask: [B, 256x256, 313] --> [B*num_windows, winsize*winsize+313, winsize*winsize+313]
        B, N_pixel, n_cls = mask.size()
        ori_h = ori_w = int(np.sqrt(N_pixel))       # 256
        patch_h = patch_w = int(np.sqrt(N_patch))       # 16/32/64/128/256
        patch_size = ori_h // patch_h       # 16/8/4/2/1
        mask = mask.view(B, patch_h, patch_size, patch_w, patch_size, n_cls).permute(0, 1, 3, 2, 4, 5).contiguous()
        mask = mask.view(B, patch_h, patch_w, patch_size*patch_size, n_cls)       # [B, p_h, p_w, psxps, 313]
        mask = torch.sum(mask, dim=-2)       # [B, p_h, p_w, 313]
        if self.num_windows == 1:
            mask = mask.view(B, N_patch, n_cls)   # [B, N, 313]
            patch_ones = torch.ones(B, N_patch, N_patch).cuda()
            color_ones = torch.ones(B, n_cls, n_cls).cuda()
            process_mask_a = torch.cat((patch_ones, mask), dim=-1)      # [B, N, N+313]
            process_mask_b = torch.cat((mask.transpose(1, 2), color_ones), dim=-1)      # [B, 313, N+313]
            process_mask = torch.cat((process_mask_a, process_mask_b), dim=1)             # [B, N+313, N+313]
            return process_mask

        mask = window_partition(mask, self.window_size)     # [B * num_windows, win_size, win_size, 313]
        mask = mask.view(-1, self.window_size * self.window_size, n_cls)        # [B*num_windows, win-size * winsize, 313]
        N_window_tokens = self.window_size * self.window_size

        patch_ones = torch.ones(B * self.num_windows, N_window_tokens, N_window_tokens).cuda()
        color_ones = torch.ones(B*self.num_windows, n_cls, n_cls).cuda()
        process_mask_a = torch.cat((patch_ones, mask), dim=-1)              # [B*num, 16x16, 16x16+313]
        process_mask_b = torch.cat((mask.transpose(1, 2), color_ones), dim=-1)      # [B*num, 313, 16x16+313]
        process_mask = torch.cat((process_mask_a, process_mask_b), dim=1)       # [B*num_wins, 16*16+313, 16*16+313]

        return process_mask

    def forward(self, x, mask):
        B, N, C = x.size()      # N = patch_tokens + color_tokens
        N_patch = N - self.n_cls
        H = W = int(np.sqrt(N_patch))
        x_norm = self.norm1(x)           # LayerNorm

        patch_tokens = x_norm[:, :-self.n_cls, :]        # [B, 16x16, C]
        if self.num_windows > 1:
            patch_tokens = patch_tokens.view(B, H, W, C)
            patch_tokens = window_partition(patch_tokens, self.window_size)
            patch_tokens = patch_tokens.view(-1, self.window_size * self.window_size, C)    # [B * num_windows, w_size * w_size, C]

        color_tokens = x_norm[:, -self.n_cls:, :]        # [B, 313, C]
        attn_mask = self.process_mask(mask, N_patch)
        patch_tokens, color_tokens = self.attn(patch_tokens, color_tokens, attn_mask)

        if self.num_windows >1:
            patch_tokens = patch_tokens.view(-1, self.window_size, self.window_size, C)
            patch_tokens = window_reverse(patch_tokens, self.window_size, H, W).view(B, N_patch, C)
        y = torch.cat((patch_tokens, color_tokens), dim=1)
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        del attn_mask

        return x



######### use color tokens as condition ######################################
class Decoder_Block_Color(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.heads = heads
        self.attn = MultiheadAttention(dim, heads, dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, color_emb, mask=None):
        x = self.norm1(x)
        tokens = torch.cat((x, color_emb), dim=1)
        y = self.attn(query=x,
                      key=tokens,
                      value=tokens,
                      attn_mask=mask)[0]
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MultiheadAttention(nn.Module):         # for vit attention
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.attn = None

        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, query, key, value, attn_mask=None):
        B, N, C = query.shape
        L = key.shape[1]

        q = self.W_q(query).reshape(B, N, self.heads, C//self.heads).permute(0, 2, 1, 3)
        k = self.W_k(key).reshape(B, L, self.heads, C//self.heads).permute(0, 2, 1, 3)
        v = self.W_v(value).reshape(B, L, self.heads, C//self.heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            expand_mask = attn_mask.unsqueeze(1).repeat(1, self.heads, 1, 1)     # [B, heads, N, L]
            attn = attn.masked_fill(expand_mask == 0, -float('inf'))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        return out, attn
