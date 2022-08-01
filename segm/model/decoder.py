import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np

from timm.models.layers import trunc_normal_

from segm.model.blocks import Block, FeedForward, Decoder_Block, Decoder_Block_Color, Multiscale_Block
from segm.model.utils import init_weights, CIELAB
from segm.engine import functional_conv2d

def bicubic_upsample(x, H, W, scaler=2):
    B, N, C = x.size()
    assert N == H*W
    x = x.permute(0, 2, 1)
    x = x.view(-1, C, H, W)
    x = nn.functional.interpolate(x, scale_factor=scaler, mode='bicubic')
    B, C, H, W = x.size()
    x = x.view(-1, C, H*W)
    x = x.permute(0,2,1)
    return x, H, W

def updown(x, H, W):
    B, N, C = x.size()
    assert N == H*W
    x = x.permute(0, 2, 1)
    x = x.view(-1, C, H, W)
    x = nn.functional.interpolate(x, scale_factor=4, mode='bicubic')
    x = nn.AvgPool2d(4)(x)
    B, C, H, W = x.size()
    x = x.view(-1, C, H*W)
    x = x.permute(0,2,1)
    return x, H, W

def pixel_upsample(x, H, W):
    B, N, C = x.size()
    assert N == H*W
    x = x.permute(0, 2, 1)
    x = x.view(-1, C, H, W)
    x = nn.PixelShuffle(2)(x)       # upsample the resolution and downscale the feature dimension.
    B, C, H, W = x.size()
    x = x.view(-1, C, H*W)
    x = x.permute(0,2,1)
    return x, H, W


def conv3x3(in_planes, out_planes, stride=1, bias=True):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1, bias=True):
    """3x3 convolution with padding, batch normalization and relu"""
    block = nn.Sequential(
        conv3x3(in_planes, out_planes, stride, bias),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True)
    )
    return block

def conv3x3_relu(in_planes, out_planes, stride=1, bias=True):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes, stride, bias),
        nn.ReLU(inplace=True)
    )
    return block


class DecoderLinear(nn.Module):
    def __init__(self, n_cls, patch_size, d_encoder):
        super().__init__()

        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_cls = n_cls

        self.head = nn.Linear(self.d_encoder, n_cls)
        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, x, im_size):
        H, W = im_size
        GS = H // self.patch_size
        x = self.head(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=GS)

        return x


class PosCNN(nn.Module):
    def __init__(self, in_chans, embed_dim, q_to_ab=None, s=1):
        super(PosCNN, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim), )
        self.s = s
        self.q_to_ab = q_to_ab      # [313, 2]

    def forward(self, x, H, W):
        B, N, C = x.shape           # [B, 313, C]
        if self.q_to_ab is not None:        # color pos.
            cnn_feat = torch.zeros(B, H, W, C).to(x.device)    # [b, 23, 23, c]
            bin = 10
            torch_ab = torch.from_numpy(self.q_to_ab).to(x.device)
            # new_ab = (torch_ab + 110) // bin        # [313, 2]
            new_ab = torch.div(torch_ab + 110, bin, rounding_mode='floor')
            cnn_feat[:, new_ab[:, 0].long(), new_ab[:, 1].long(), :] = x      # [B, N, C]

            conv_cnn_feat = self.proj(cnn_feat.permute(0, 3, 1, 2))     # [B, C, 23, 23]
            conv_cnn_feat = conv_cnn_feat.permute(0, 2, 3, 1)       # [B, 23, 23, C]
            x_pos = torch.zeros_like(x)
            x_pos[:, :, :] = conv_cnn_feat[:, new_ab[:, 0].long(), new_ab[:, 1].long(), :]     # [B, N, C]
            x = x + x_pos
        else:       # patch pos.
            feat_token = x
            cnn_feat = feat_token.transpose(1, 2).contiguous().view(B, C, H, W)
            x = self.proj(cnn_feat) + cnn_feat
            x = x.flatten(2).transpose(1, 2)
        return x

    def no_weight_decay(self):
        return ['proj.%d.weight' % i for i in range(4)]


class Sin_Pos(nn.Module):
    def __init__(self, d_model, dimension):
        super(Sin_Pos, self).__init__()
        self.dimension = dimension
        self.d_model = d_model

    def forward(self, x):
        if self.dimension == 1:
            # for color tokens.
            B, N, C = x.shape
            pe = torch.zeros(N, C).to(x.device)     # [N, C]
            position = torch.arange(0, N).unsqueeze(1)      # [N, 1]
            div_term = torch.exp((torch.arange(0, C, dtype=torch.float) * -(math.log(10000.0) / C)))
            pe[:, 0::2] = torch.sin(position.float() * div_term)
            pe[:, 1::2] = torch.cos(position.float() * div_term)
            pe = pe.unsqueeze(0).repeat(B)      #[B, N, C]
            x += pe
        elif self.dimension == 2:
            B, N, C = x.shape
            H = W = math.sqrt(N)        #16
            patch_token = x.transpose(1, 2).contiguous().view(B, C, H, W)
            pe = torch.zeros(C, H, W).to(x.device)
            d_model = C //2
            div_term = torch.exp(torch.arange(0, d_model, 2) * -math.log(10000.0) / d_model)
            pos_w = torch.arange(0., W).unsqueeze(1)
            pos_h = torch.arange(0., H).unsqueeze(1)
            pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
            pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
            pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
            pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
            pe = pe.unsqueeze(0).repeat(B)
            patch_token += pe
            x = patch_token.flatten(2).transpose(1, 2)     # [B, N, C]
        return x


# Upsample Block
class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel, upscale=2):
        super(Upsample, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=upscale, stride=upscale),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.deconv(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        return out


class MaskTransformer(nn.Module):
    def __init__(
        self,
        n_cls,
        patch_size,
        d_encoder,
        n_layers,
        n_heads,
        d_model,
        d_ff,
        drop_path_rate,
        dropout,
        add_l1_loss,
        color_position,
        change_mask,
        color_as_condition,
        multi_scaled,
        crop_size,
        downchannel,
        add_conv,
        before_classify,
        l1_conv,
        l1_linear,
        add_edge,
        n_blocks,
        without_colorattn,
        without_colorquery,
        without_classification,
        sin_color_pos,
    ):
        super().__init__()
        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.n_cls = n_cls
        self.d_model = d_model
        self.d_ff = d_ff
        self.scale = d_model ** -0.5
        self.add_l1_loss = add_l1_loss
        self.color_position = color_position
        self.change_mask = change_mask
        self.color_as_condition = color_as_condition
        self.multi_scaled = multi_scaled
        self.downchannel = downchannel
        self.add_conv = add_conv
        self.before_classify = before_classify
        self.l1_conv = l1_conv
        self.l1_linear = l1_linear
        self.add_edge = add_edge
        self.n_blocks = n_blocks
        self.without_colorattn = without_colorattn
        self.without_colorquery = without_colorquery
        self.without_classification = without_classification
        self.sin_color_pos = sin_color_pos
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.per_block = nn.ModuleList(
            [Decoder_Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
        )

        self.blocks = nn.ModuleList([self.per_block for i in range(n_blocks)])

        if self.color_position:
            self.cielab = CIELAB()
            self.q_to_ab = self.cielab.q_to_ab
            self.cls_emb = nn.Parameter(torch.randn(1, n_cls, d_model))
            if self.sin_color_pos:
                self.pos_color = Sin_Pos(d_model, dimension=1)
                self.pos_patch = Sin_Pos(d_model, dimension=2)
            else:
                self.pos_color = PosCNN(d_model, d_model, self.q_to_ab)
                self.pos_patch = PosCNN(d_model, d_model)
        else:
            self.cls_emb = nn.Parameter(torch.randn(1, n_cls, d_model))     # all learnable
        self.proj_patch = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        self.proj_classes = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        self.decoder_norm = nn.LayerNorm(d_model)
        if self.add_conv:
            self.conv_layers = nn.ModuleList(
                [nn.Conv2d(d_model, d_model, kernel_size=3, stride=1, padding=1) for i in range(n_blocks)]
            )
            self.color_linear = nn.ModuleList(
                [nn.Linear(d_model, d_model) for i in range(n_blocks)]
            )
            self.upsampler = nn.Sequential(
                conv3x3_bn_relu(d_model, d_model),
                nn.ConvTranspose2d(d_model, d_model, kernel_size=4, stride=2, padding=1, bias=True),
                nn.ReLU(True),
                conv3x3_bn_relu(d_model, d_model),
                nn.ConvTranspose2d(d_model, d_model, kernel_size=4, stride=2, padding=1, bias=True),
                nn.ReLU(True),
                conv3x3_bn_relu(d_model, d_model),
                nn.ConvTranspose2d(d_model, d_model, kernel_size=4, stride=2, padding=1, bias=True),
                nn.ReLU(True),
                conv3x3_bn_relu(d_model, d_model),
                nn.ConvTranspose2d(d_model, d_model, kernel_size=4, stride=2, padding=1, bias=True),
                )

        if self.add_edge:
            self.conv_edge = nn.Conv2d(d_model * 2, d_model, kernel_size=3, stride=1, padding=1, bias=True)

        self.proj_dec = nn.Linear(d_encoder, d_model)

        if self.add_l1_loss:
            if self.l1_conv:
                if d_model == 192:      # tiny
                    self.upsampler_l1 = nn.Sequential(
                        conv3x3_bn_relu(d_model, d_model),
                        nn.ConvTranspose2d(d_model, d_model // 2, kernel_size=4, stride=2, padding=1, bias=True),
                        nn.ReLU(True),
                        conv3x3_bn_relu(d_model // 2, d_model // 2),
                        nn.ConvTranspose2d(d_model // 2, d_model // 4, kernel_size=4, stride=2, padding=1, bias=True),
                        nn.ReLU(True),
                        conv3x3_bn_relu(d_model // 4, d_model // 4),
                        nn.ConvTranspose2d(d_model // 4, d_model // 8, kernel_size=4, stride=2, padding=1, bias=True),
                        nn.ReLU(True),
                        conv3x3_bn_relu(d_model // 8, d_model // 8),
                        nn.ConvTranspose2d(d_model // 8, d_model // 8, kernel_size=4, stride=2, padding=1, bias=True),
                        conv3x3(d_model // 8, 2)
                    )
                elif d_model == 384 or d_model == 768:        # small
                    self.upsampler_l1 = nn.Sequential(
                        conv3x3_bn_relu(d_model, d_model),
                        nn.ConvTranspose2d(d_model, d_model // 2, kernel_size=4, stride=2, padding=1, bias=True),
                        nn.ReLU(True),
                        conv3x3_bn_relu(d_model // 2, d_model // 2),
                        nn.ConvTranspose2d(d_model // 2, d_model // 4, kernel_size=4, stride=2, padding=1, bias=True),
                        nn.ReLU(True),
                        conv3x3_bn_relu(d_model // 4, d_model // 4),
                        nn.ConvTranspose2d(d_model // 4, d_model // 8, kernel_size=4, stride=2, padding=1, bias=True),
                        nn.ReLU(True),
                        conv3x3_bn_relu(d_model // 8, d_model // 8),
                        nn.ConvTranspose2d(d_model // 8, d_model // 16, kernel_size=4, stride=2, padding=1, bias=True),
                        conv3x3(d_model // 16, 2))
                elif d_model == 1024:       # Large or base
                    self.upsampler_l1 = nn.Sequential(
                        conv3x3_bn_relu(d_model, d_model),
                        nn.ConvTranspose2d(d_model, d_model // 4, kernel_size=4, stride=2, padding=1, bias=True),
                        nn.ReLU(True),
                        conv3x3_bn_relu(d_model // 4, d_model // 4),
                        nn.ConvTranspose2d(d_model // 4, d_model // 16, kernel_size=4, stride=2, padding=1, bias=True),
                        nn.ReLU(True),
                        conv3x3_bn_relu(d_model // 16, d_model // 16),
                        nn.ConvTranspose2d(d_model // 16, d_model // 64, kernel_size=4, stride=2, padding=1, bias=True),
                        nn.ReLU(True),
                        conv3x3_bn_relu(d_model // 64, d_model // 64),
                        nn.ConvTranspose2d(d_model // 64, d_model // 256, kernel_size=4, stride=2, padding=1,
                                           bias=True),
                        conv3x3(d_model // 256, 2)
                    )

                self.tanh = nn.Tanh()
            elif self.l1_linear:
                self.conv_out = conv3x3(d_model, 2)
                self.tanh = nn.Tanh()

        self.mask_norm = nn.LayerNorm(n_cls)
        if self.without_colorquery:
            self.classifier = nn.Linear(d_model, self.n_cls)

        self.apply(init_weights)
        trunc_normal_(self.cls_emb, std=0.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_emb"}

    def calculate_mask(self, mask):
        # mask: [B, 256x256, 313]-> [B, 16x16+313, 16x16+313]
        B, N, n_cls = mask.size()
        H = W = int(math.sqrt(N))       # H=W=256
        process_mask = mask.view(B, H//self.patch_size, self.patch_size, W//self.patch_size, self.patch_size, n_cls)
        process_mask = process_mask.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, (H//self.patch_size) * (W//self.patch_size), self.patch_size*self.patch_size, n_cls)
        process_mask = torch.sum(process_mask, dim=2)
        mask_t = process_mask.transpose(1, 2)   # [B, 313, 16x16]
        mask_p = torch.ones((B, H//self.patch_size* W//self.patch_size, H//self.patch_size* W//self.patch_size)).to(process_mask.device)
        mask_c = torch.ones(B, n_cls, n_cls).to(process_mask.device)
        mask_p = torch.cat((mask_p, process_mask), dim=-1)   # [B, 16x16, 16x16+313]
        mask_c = torch.cat((mask_t, mask_c), dim=-1)    # [B, 313, 16x16+313]
        process_mask = torch.cat((mask_p, mask_c), dim=1)       # [B, 16x16+313, 16x16+313]
        return process_mask

    def forward(self, x, im_size, input_mask=None, patch_pos=None, img_l=None):
        H, W = im_size
        GS = H // self.patch_size
        B = x.size(0)
        x = self.proj_dec(x)
        if self.color_position:
            if self.sin_color_pos:
                cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
                cls_emb = self.pos_color(cls_emb)
                x = self.pos_patch(x)
            else:
                pos_h, pos_w = 23, 23
                cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
                cls_emb = self.pos_color(cls_emb, pos_h, pos_w)     # cpvt for color tokens.
                x = self.pos_patch(x, GS, GS)       # cpvt for patch tokens.
        else:
            cls_emb = self.cls_emb.expand(x.size(0), -1, -1)

        x = torch.cat((x, cls_emb), 1)
        if input_mask is not None:
            process_mask = self.calculate_mask(input_mask)
        else:
            process_mask = None
        for block_idx in range(self.n_blocks):
            for layer_idx in range(self.n_layers):
                x = self.blocks[block_idx][layer_idx](x, mask=process_mask, without_colorattn=self.without_colorattn)

            if self.add_conv:       # 16x16-cls && 256x256-regression, conv after trans.
                patch, color = x[:, :-self.n_cls], x[:, -self.n_cls: ]
                patch_h = patch_w = int(math.sqrt(patch.size(1)))
                patch = patch.contiguous().view(B, patch_h, patch_w, self.d_model).permute(0, 3, 1, 2)  # [B, 192, h, w]
                patch = self.conv_layers[block_idx](patch).contiguous()      # conv after per transformer block for patch.
                color = self.color_linear[block_idx](color)     # linear after per transformer blocks for color.
                patch = patch.view(B, self.d_model, patch_h * patch_w).transpose(1, 2)
                x = torch.cat((patch, color), dim=1)

        x = self.decoder_norm(x)

        down_patches, cls_seg_feat = x[:, : -self.n_cls], x[:, -self.n_cls :]
        if self.add_conv and not self.without_classification:       # default.
            patches = down_patches.contiguous().view(B, GS, GS, self.d_model).permute(0, 3, 1, 2)
            patches = self.upsampler(patches).contiguous()   # [B, 192, 256, 256]
            patches = patches.view(B, self.d_model, H*W).transpose(1, 2).contiguous()    # [B, 256x256, 192]

        if self.add_l1_loss:
            if self.before_classify:
                reshape_patch = patches.view(B, GS, GS, self.d_model).permute(0, 3, 1, 2)
                out_feature = self.upsampler(reshape_patch)     # [B, 2, 256, 256]
                out_feature = nn.Tanh()(out_feature)        # normalized to [-1, 1]
            elif self.l1_conv:
                down_patches = down_patches.contiguous().view(B, GS, GS, self.d_model).permute(0, 3, 1, 2)
                out_feature = self.upsampler_l1(down_patches)       # [B, 192, 16, 16]-> [B, 2, 256, 256]
                out_feature = self.tanh(out_feature)      # normalized to [-1, 1]
            elif self.l1_linear:
                reshape_patch = patches.transpose(1, 2).contiguous().view(B, self.d_model, H, W)
                out_feature = self.conv_out(reshape_patch)      # [B, 2, H, W]
                out_feature = self.tanh(out_feature)        # normalized to [-1, 1]
        else:
            out_feature = None

        if not self.without_colorquery and not self.without_classification:     # default.
            patches = patches @ self.proj_patch
            cls_seg_feat = cls_seg_feat @ self.proj_classes

            patches = patches / patches.norm(dim=-1, keepdim=True)
            cls_seg_feat = cls_seg_feat / cls_seg_feat.norm(dim=-1, keepdim=True)

            masks = patches @ cls_seg_feat.transpose(1, 2)      # [B, 16x16, 313]
            masks = self.mask_norm(masks)       # [B, N, 313]
            if input_mask is not None:  # add_mask == True
                if self.multi_scaled or self.add_conv:
                    new_mask = input_mask  # [B, 256x256, 313]
                else:
                    new_mask = process_mask[:, :-self.n_cls, -self.n_cls:]  # [B, N, 313]
                masks = masks.masked_fill(new_mask == 0, -float('inf'))  # is it right???
        elif self.without_colorquery:
            assert self.without_colorquery is True
            masks = self.classifier(patches)    # [B, 256x256, 192]-> [B, 256x256, 313]
        elif self.without_classification:
            assert self.without_classification is True
            return None, out_feature

        if self.multi_scaled or self.add_conv:
            masks = rearrange(masks, "b (h w) n -> b n h w", h=H)       # [B, 313, 256, 256]

        return masks, out_feature

    def get_attention_map(self, x, layer_id):
        if layer_id >= self.n_layers or layer_id < 0:
            raise ValueError(
                f"Provided layer_id: {layer_id} is not valid. 0 <= {layer_id} < {self.n_layers}."
            )
        x = self.proj_dec(x)
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls_emb), 1)
        for i, blk in enumerate(self.blocks):
            if i < layer_id:
                x = blk(x)
            else:
                return blk(x, return_attention=True)
