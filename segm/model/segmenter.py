import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from segm.model.utils import padding, unpadding, SoftEncodeAB, CIELAB, AnnealedMeanDecodeQ
from timm.models.layers import trunc_normal_
import time


class GetClassWeights:
    def __init__(self, cielab, lambda_=0.5, device='cuda'):
        prior = torch.from_numpy(cielab.gamut.prior)

        uniform = torch.zeros_like(prior)
        uniform[prior > 0] = 1 / (prior > 0).sum().type_as(uniform)

        self.weights = 1 / ((1 - lambda_) * prior + lambda_ * uniform)
        self.weights /= torch.sum(prior * self.weights)

    def __call__(self, ab_actual):
        return self.weights[ab_actual.argmax(dim=1, keepdim=True)].to(ab_actual.device)


class RebalanceLoss(Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, data_input, weights):
        ctx.save_for_backward(weights)

        return data_input.clone()

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        weights, = ctx.saved_tensors

        # reweigh gradient pixelwise so that rare colors get a chance to
        # contribute
        grad_input = grad_output * weights

        # second return value is None since we are not interested in the
        # gradient with respect to the weights
        return grad_input, None


class Segmenter(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        n_cls,
        before_classify,
        backbone,
        without_classification,
    ):
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = encoder.patch_size
        self.encoder = encoder
        self.decoder = decoder
        self.default_cielab = CIELAB()
        self.encode_ab = SoftEncodeAB(self.default_cielab)
        self.decode_q = AnnealedMeanDecodeQ(self.default_cielab, T=0.38)
        self.class_rebal_lambda = 0.5
        self.get_class_weights = GetClassWeights(self.default_cielab,
                                                 lambda_=self.class_rebal_lambda)
        self.before_classify = before_classify
        self.backbone = backbone
        self.without_classification = without_classification
        self.rebalance_loss = RebalanceLoss.apply

    @torch.jit.ignore
    def no_weight_decay(self):
        def append_prefix_no_weight_decay(prefix, module):
            return set(map(lambda x: prefix + x, module.no_weight_decay()))

        nwd_params = append_prefix_no_weight_decay("encoder.", self.encoder).union(
            append_prefix_no_weight_decay("decoder.", self.decoder)
        )
        return nwd_params


    def normalize_l(self, l, to):
        # follow Real-Time/ CIC
        normalized = (l-50)/100.
        return normalized

    @torch.cuda.amp.autocast()
    def forward(self, l, gt_ab, input_mask=None):
        im = self.normalize_l(l, (-1, 1))  # [-1, 1]

        im = im.repeat(1, 3, 1, 1)
        H, W = im.size(2), im.size(3)
        x, x_pos = self.encoder(im, return_features=True)      #x: [BS, N, D]
        if 'vit' in self.backbone:
            # remove CLS/DIST tokens for decoding
            num_extra_tokens = 1 + self.encoder.distilled
            x = x[:, num_extra_tokens:]
        if x_pos is not None:
            x_pos = x_pos[:, num_extra_tokens:]
        masks, out_feat = self.decoder(x, (H, W), input_mask, x_pos, im)     # [b, 313, H/P, W/P], out_feat: [B, 2, h, w]

        if not self.without_classification:
            q_pred = masks      # multi-scaled, [B, 313, 256, 256]
            q_actual = self.encode_ab(gt_ab)
            # rebalancing
            color_weights = self.get_class_weights(q_actual)
            q_pred = self.rebalance_loss(q_pred, color_weights)
            ab_pred = self.decode_q(q_pred)  # softmax to [0, 1]
        else:
            ab_pred, q_pred, q_actual = None, None, None
        return ab_pred, q_pred, q_actual, out_feat

    def inference(self, l, img_ab, input_mask=None, appli=False):
        im = self.normalize_l(l, (-1, 1))       # [-1, 1]
        im = im.repeat(1, 3, 1, 1)
        H, W = im.size(2), im.size(3)

        x, x_pos = self.encoder(im, return_features=True)  # x: [BS, N, D]
        if 'vit' in self.backbone:
            # remove CLS/DIST tokens for decoding
            num_extra_tokens = 1 + self.encoder.distilled
            x = x[:, num_extra_tokens:]
        if x_pos is not None:
            x_pos = x_pos[:, num_extra_tokens:]
        masks, out_feat = self.decoder(x, (H, W), input_mask, x_pos, im)  # [b, K, H/P, W/P]
        if not self.without_classification:
            q_pred = masks      # [B, 313, 256, 256]
            ab_pred = self.decode_q(q_pred, appli=appli)     # softmax to [0, 1]
            q_actual = self.encode_ab(img_ab)
        else:
            ab_pred, q_pred, q_actual = None, None, None

        return ab_pred, q_pred, q_actual, out_feat

    def convert_ab(self, img_ab):
        q_actual = self.encode_ab(img_ab)
        ab_actual = self.decode_q(q_actual, is_actual=True)
        return ab_actual

    def get_attention_map_enc(self, im, layer_id):
        return self.encoder.get_attention_map(im, layer_id)

    def get_attention_map_dec(self, im, layer_id):
        x = self.encoder(im, return_features=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]

        return self.decoder.get_attention_map(x, layer_id)
