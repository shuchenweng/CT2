import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import defaultdict
import warnings
from skimage import color, io, transform
import os
from timm.models.layers import trunc_normal_
import torchvision.models as models

import segm.utils.torch as ptu
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import glob


def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


def resize_pos_embed(posemb, grid_old_shape, grid_new_shape, num_extra_tokens):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    posemb_tok, posemb_grid = (
        posemb[:, :num_extra_tokens],
        posemb[0, num_extra_tokens:],
    )
    if grid_old_shape is None:
        gs_old_h = int(math.sqrt(len(posemb_grid)))
        gs_old_w = gs_old_h
    else:
        gs_old_h, gs_old_w = grid_old_shape

    gs_h, gs_w = grid_new_shape
    posemb_grid = posemb_grid.reshape(1, gs_old_h, gs_old_w, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_h, gs_w), mode="bilinear")        # ??
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_h * gs_w, -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if "model" in state_dict:
        # For deit models
        state_dict = state_dict["model"]
    num_extra_tokens = 1 + ("dist_token" in state_dict.keys())
    patch_size = model.patch_size
    image_size = model.patch_embed.image_size
    for k, v in state_dict.items():
        if k == "pos_embed" and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(
                v,
                None,
                (image_size[0] // patch_size, image_size[1] // patch_size),
                num_extra_tokens,
            )
        out_dict[k] = v
    return out_dict


def padding(im, patch_size, fill_value=0):
    # make the image sizes divisible by patch_size
    H, W = im.size(2), im.size(3)
    pad_h, pad_w = 0, 0
    if H % patch_size > 0:
        pad_h = patch_size - (H % patch_size)
    if W % patch_size > 0:
        pad_w = patch_size - (W % patch_size)
    im_padded = im
    if pad_h > 0 or pad_w > 0:
        im_padded = F.pad(im, (0, pad_w, 0, pad_h), value=fill_value)
    return im_padded


def unpadding(y, target_size):
    H, W = target_size
    H_pad, W_pad = y.size(2), y.size(3)
    # crop predictions on extra pixels coming from padding
    extra_h = H_pad - H
    extra_w = W_pad - W
    if extra_h > 0:
        y = y[:, :, :-extra_h]
    if extra_w > 0:
        y = y[:, :, :, :-extra_w]
    return y


def resize(im, smaller_size):
    h, w = im.shape[2:]
    if h < w:
        ratio = w / h
        h_res, w_res = smaller_size, ratio * smaller_size
    else:
        ratio = h / w
        h_res, w_res = ratio * smaller_size, smaller_size
    if min(h, w) < smaller_size:
        im_res = F.interpolate(im, (int(h_res), int(w_res)), mode="bilinear")
    else:
        im_res = im
    return im_res


def sliding_window(im, flip, window_size, window_stride):
    B, C, H, W = im.shape
    ws = window_size

    windows = {"crop": [], "anchors": []}
    h_anchors = torch.arange(0, H, window_stride)
    w_anchors = torch.arange(0, W, window_stride)
    h_anchors = [h.item() for h in h_anchors if h < H - ws] + [H - ws]
    w_anchors = [w.item() for w in w_anchors if w < W - ws] + [W - ws]
    for ha in h_anchors:
        for wa in w_anchors:
            window = im[:, :, ha : ha + ws, wa : wa + ws]
            windows["crop"].append(window)
            windows["anchors"].append((ha, wa))
    windows["flip"] = flip
    windows["shape"] = (H, W)
    return windows


def merge_windows(windows, window_size, ori_shape):
    ws = window_size
    im_windows = windows["seg_maps"]
    anchors = windows["anchors"]
    C = im_windows[0].shape[0]
    H, W = windows["shape"]
    flip = windows["flip"]

    logit = torch.zeros((C, H, W), device=im_windows.device)
    count = torch.zeros((1, H, W), device=im_windows.device)
    for window, (ha, wa) in zip(im_windows, anchors):
        logit[:, ha : ha + ws, wa : wa + ws] += window
        count[:, ha : ha + ws, wa : wa + ws] += 1
    logit = logit / count
    logit = F.interpolate(
        logit.unsqueeze(0),
        ori_shape,
        mode="bilinear",
    )[0]
    if flip:
        logit = torch.flip(logit, (2,))
    result = F.softmax(logit, 0)
    return result


def inference(
    model,
    ims,
    ims_metas,
    ori_shape,
    window_size,
    window_stride,
    batch_size,
):
    C = model.n_cls
    seg_map = torch.zeros((C, ori_shape[0], ori_shape[1]), device=ptu.device)
    for im, im_metas in zip(ims, ims_metas):
        im = im.to(ptu.device)
        im = resize(im, window_size)
        flip = im_metas["flip"]
        windows = sliding_window(im, flip, window_size, window_stride)
        crops = torch.stack(windows.pop("crop"))[:, 0]
        B = len(crops)
        WB = batch_size
        seg_maps = torch.zeros((B, C, window_size, window_size), device=im.device)
        with torch.no_grad():
            for i in range(0, B, WB):
                seg_maps[i : i + WB] = model.forward(crops[i : i + WB])
        windows["seg_maps"] = seg_maps
        im_seg_map = merge_windows(windows, window_size, ori_shape)
        seg_map += im_seg_map
    seg_map /= len(ims)
    return seg_map


def num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    n_params = sum([torch.prod(torch.tensor(p.size())) for p in model_parameters])
    return n_params.item()



#########  encode AB ###############
class SoftEncodeAB:
    def __init__(self, cielab, neighbours=5, sigma=5.0, device='cuda'):
        self.cielab = cielab
        self.q_to_ab = torch.from_numpy(self.cielab.q_to_ab).to(device)

        self.neighbours = neighbours
        self.sigma = sigma

    def __call__(self, ab):
        n, _, h, w = ab.shape

        m = n * h * w

        # find nearest neighbours
        ab_ = ab.permute(1, 0, 2, 3).reshape(2, -1)
        q_to_ab = self.q_to_ab.type(ab_.dtype)

        cdist = torch.cdist(q_to_ab, ab_.t())

        nns = cdist.argsort(dim=0)[:self.neighbours, :]

        # gaussian weighting
        nn_gauss = ab.new_zeros(self.neighbours, m)

        for i in range(self.neighbours):
            nn_gauss[i, :] = self._gauss_eval(
                q_to_ab[nns[i, :], :].t(), ab_, self.sigma)

        nn_gauss /= nn_gauss.sum(dim=0, keepdim=True)

        # expand
        bins = self.cielab.gamut.EXPECTED_SIZE

        q = ab.new_zeros(bins, m)

        q[nns, torch.arange(m).repeat(self.neighbours, 1)] = nn_gauss
        # return: [bs, 313, 256, 256]
        return q.reshape(bins, n, h, w).permute(1, 0, 2, 3)

    @staticmethod
    def _gauss_eval(x, mu, sigma):
        norm = 1 / (2 * math.pi * sigma)

        return norm * torch.exp(-torch.sum((x - mu)**2, dim=0) / (2 * sigma**2))


########### CIELAB #####################################
# _SOURCE_DIR = os.path.abspath(os.path.dirname(__file__))
_SOURCE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))       # last dir
_RESOURCE_DIR = os.path.join(_SOURCE_DIR, 'resources')

def lab_to_rgb(img):
    assert img.dtype == np.float32

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        return (255 * np.clip(color.lab2rgb(img), 0, 1)).astype(np.uint8)

def get_resource_path(path):
   return os.path.join(_RESOURCE_DIR, path)


class ABGamut:
    RESOURCE_POINTS = get_resource_path('ab-gamut.npy')
    RESOURCE_PRIOR = get_resource_path('q-prior.npy')

    DTYPE = np.float32
    EXPECTED_SIZE = 313

    def __init__(self):
        self.points = np.load(self.RESOURCE_POINTS, allow_pickle=True).astype(self.DTYPE)
        self.prior = np.load(self.RESOURCE_PRIOR, allow_pickle=True).astype(self.DTYPE)

        assert self.points.shape == (self.EXPECTED_SIZE, 2)
        assert self.prior.shape == (self.EXPECTED_SIZE,)


class CIELAB:
    L_MEAN = 50

    AB_BINSIZE = 10
    AB_RANGE = [-110 - AB_BINSIZE // 2, 110 + AB_BINSIZE // 2, AB_BINSIZE]
    AB_DTYPE = np.float32

    Q_DTYPE = np.int64

    RGB_RESOLUTION = 101
    RGB_RANGE = [0, 1, RGB_RESOLUTION]
    RGB_DTYPE = np.float64

    def __init__(self, gamut=None):
        self.gamut = gamut if gamut is not None else ABGamut()

        a, b, self.ab = self._get_ab()

        self.ab_gamut_mask = self._get_ab_gamut_mask(
            a, b, self.ab, self.gamut)

        self.ab_to_q = self._get_ab_to_q(self.ab_gamut_mask)
        self.q_to_ab = self._get_q_to_ab(self.ab, self.ab_gamut_mask)

    @classmethod
    def _get_ab(cls):
        a = np.arange(*cls.AB_RANGE, dtype=cls.AB_DTYPE)
        b = np.arange(*cls.AB_RANGE, dtype=cls.AB_DTYPE)

        b_, a_ = np.meshgrid(a, b)
        ab = np.dstack((a_, b_))

        return a, b, ab

    @classmethod
    def _get_ab_gamut_mask(cls, a, b, ab, gamut):
        ab_gamut_mask = np.full(ab.shape[:-1], False, dtype=bool)

        a = np.digitize(gamut.points[:, 0], a) - 1
        b = np.digitize(gamut.points[:, 1], b) - 1

        for a_, b_ in zip(a, b):
            ab_gamut_mask[a_, b_] = True

        return ab_gamut_mask

    @classmethod
    def _get_ab_to_q(cls, ab_gamut_mask):
        ab_to_q = np.full(ab_gamut_mask.shape, -1, dtype=cls.Q_DTYPE)

        ab_to_q[ab_gamut_mask] = np.arange(np.count_nonzero(ab_gamut_mask))

        return ab_to_q

    @classmethod
    def _get_q_to_ab(cls, ab, ab_gamut_mask):
        return ab[ab_gamut_mask] + cls.AB_BINSIZE / 2

    @classmethod
    def _plot_ab_matrix(cls, mat, pixel_borders=False, ax=None, title=None):
        if ax is None:
            _, ax = plt.subplots()

        imshow = partial(ax.imshow,
                         np.flip(mat, axis=0),
                         extent=[*cls.AB_RANGE[:2]] * 2)

        if len(mat.shape) < 3 or mat.shape[2] == 1:
            im = imshow(cmap='jet')

            fig = plt.gcf()
            fig.colorbar(im, cax=fig.add_axes())
        else:
            imshow()

        # set title
        if title is not None:
            ax.set_title(title)

        # set axes labels
        ax.set_xlabel("$b$")
        ax.set_ylabel("$a$")

        # minor ticks
        tick_min_minor = cls.AB_RANGE[0]
        tick_max_minor = cls.AB_RANGE[1]

        if pixel_borders:

            ax.set_xticks(
                np.linspace(tick_min_minor, tick_max_minor, mat.shape[1] + 1),
                minor=True)

            ax.set_yticks(
                np.linspace(tick_min_minor, tick_max_minor, mat.shape[0] + 1),
                minor=True)

            ax.grid(which='minor',
                    color='w',
                    linestyle='-',
                    linewidth=2)

        # major ticks
        tick_min_major = tick_min_minor + cls.AB_BINSIZE // 2
        tick_max_major = tick_max_minor - cls.AB_BINSIZE // 2

        ax.set_xticks(np.linspace(tick_min_major, tick_max_major, 5))
        ax.set_yticks(np.linspace(tick_min_major, tick_max_major, 5))

        # some of this will be obscured by the minor ticks due to a five year
        # old matplotlib bug...
        ax.grid(which='major',
                color='k',
                linestyle=':',
                dashes=(1, 4))

        # tick marks
        for ax_ in ax.xaxis, ax.yaxis:
            ax_.set_ticks_position('both')

        ax.tick_params(axis='both', which='major', direction='in')
        ax.tick_params(axis='both', which='minor', length=0)

        # limits
        lim_min = tick_min_major - cls.AB_BINSIZE
        lim_max = tick_max_major + cls.AB_BINSIZE

        ax.set_xlim([lim_min, lim_max])
        ax.set_ylim([lim_min, lim_max])

        # invert y-axis
        ax.invert_yaxis()

    def bin_ab(self, ab):
        ab_discrete = ((ab + 110) / self.AB_RANGE[2]).astype(int)

        a, b = np.hsplit(ab_discrete.reshape(-1, 2), 2)
        q = self.ab_to_q[a, b].reshape(*ab.shape[:2])

        return q

    def plot_ab_gamut(self, l=50, ax=None):
        assert l >= 50 and l <= 100

        # construct Lab color space slice for given L
        l_ = np.full(self.ab.shape[:2], l, dtype=self.ab.dtype)
        color_space_lab = np.dstack((l_, self.ab))

        # convert to RGB
        color_space_rgb = lab_to_rgb(color_space_lab)

        # mask out of gamut colors
        color_space_rgb[~self.ab_gamut_mask, :] = 255

        # display color space
        self._plot_ab_matrix(color_space_rgb,
                             pixel_borders=True,
                             ax=ax,
                             title=r"$RGB(a, b \mid L = {})$".format(l))

    def plot_empirical_distribution(self, dataset, ax=None, verbose=False):
        # accumulate ab values
        ab_acc = np.zeros([self.AB_RANGE[1] - self.AB_RANGE[0]] * 2)

        for i in range(len(dataset)):
            img = dataset[i]

            if verbose:
                fmt = "\rprocessing image {}/{}"

                print(fmt.format(i + 1, len(dataset)),
                      end=('\n' if i == len(dataset) - 1 else ''),
                      flush=True)

            img = np.moveaxis(img, 0, -1)
            ab_rounded = np.round(img[:, :, 1:].reshape(-1, 2)).astype(int)
            ab_offset = ab_rounded - self.AB_RANGE[0]

            np.add.at(ab_acc, tuple(np.split(ab_offset, 2, axis=1)), 1)

        # convert to log scale
        ab_acc[ab_acc == 0] = np.nan

        ab_acc_log = np.log10(ab_acc) - np.log10(len(dataset))

        # display distribution
        self._plot_ab_matrix(ab_acc_log, ax=ax, title=r"$log(P(a, b))$")


class AnnealedMeanDecodeQ:
    def __init__(self, cielab, T, device='cuda'):
        self.q_to_ab = torch.from_numpy(cielab.q_to_ab).to(device)

        self.T = T

    def __call__(self, q, is_actual=False, appli=False):
        if self.T == 0:
            # makeing this a special case is somewhat ugly but I have found
            # no way to make this a special case of the branch below (in
            # NumPy that would be trivial)
            ab = self._unbin(self._mode(q))
        else:
            if is_actual is False:
                q = self._annealed_softmax(q, appli=appli)

            a = self._annealed_mean(q, 0)
            b = self._annealed_mean(q, 1)
            ab = torch.cat((a, b), dim=1)

        return ab.type(q.dtype)

    def _mode(self, q):
        return q.max(dim=1, keepdim=True)[1]


    def _unbin(self, q):
        _, _, h, w = q.shape        # [bs, 1, h, w]

        ab = torch.stack([
            self.q_to_ab.index_select(
                0, q_.flatten()
            ).reshape(h, w, 2).permute(2, 0, 1)

            for q_ in q
        ])

        return ab

    def _annealed_softmax(self, q, appli=False, change_mask=None):
        q_exp = torch.exp(q / self.T)
        if not appli:
            q_softmax = q_exp / q_exp.sum(dim=1, keepdim=True)
        else:
            q_softmax = q_exp / q_exp.sum(dim=1, keepdim=True)      # [bs, 313, 256, 256]

        return q_softmax

    def _annealed_mean(self, q, d):
        am = torch.tensordot(q, self.q_to_ab[:, d], dims=((1,), (0,)))

        return am.unsqueeze(dim=1)


########################### VGG ######################################################
def load_model(model_name, model_dir):
    assert os.path.exists(model_dir)
    model = eval('models.%s(init_weights=False)' % model_name)
    path_format = os.path.join(model_dir, '%s-[a-z0-9]*.pth' % model_name)
    model_path = glob.glob(path_format)[0]
    model.load_state_dict(torch.load(model_path))
    return model


class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        model_dir = os.path.join(os.getcwd(), 'segm/resources')
        # model_dir = os.path.join(opt.pretrained_dir, 'vgg')
        model = load_model('vgg19', model_dir)
        vgg_pretrained_features = model.features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


def lab2xyz(lab):
    y_int = (lab[:, 0, :, :] + 16.) / 116.
    x_int = (lab[:, 1, :, :] / 500.) + y_int
    z_int = y_int - (lab[:, 2, :, :] / 200.)
    z_int = torch.max(torch.Tensor((0,)).to(lab.device), z_int)
    out = torch.cat((x_int[:, None, :, :], y_int[:, None, :, :], z_int[:, None, :, :]), dim=1)
    mask = (out > .2068966).type(torch.FloatTensor)
    mask = mask.to(lab.device)
    out = (out ** 3.) * mask + (out - 16. / 116.) / 7.787 * (1 - mask)
    sc = torch.Tensor((0.95047, 1., 1.08883))[None, :, None, None]
    sc = sc.to(out.device)
    out = out * sc
    return out


def xyz2rgb(xyz):
    r = 3.24048134 * xyz[:, 0, :, :] - 1.53715152 * xyz[:, 1, :, :] - 0.49853633 * xyz[:, 2, :, :]
    g = -0.96925495 * xyz[:, 0, :, :] + 1.87599 * xyz[:, 1, :, :] + .04155593 * xyz[:, 2, :, :]
    b = .05564664 * xyz[:, 0, :, :] - .20404134 * xyz[:, 1, :, :] + 1.05731107 * xyz[:, 2, :, :]

    rgb = torch.cat((r[:, None, :, :], g[:, None, :, :], b[:, None, :, :]), dim=1)
    rgb = torch.max(rgb, torch.zeros_like(rgb))
    mask = (rgb > .0031308).type(torch.FloatTensor)
    mask = mask.to(xyz.device)
    rgb = (1.055 * (rgb ** (1. / 2.4)) - 0.055) * mask + 12.92 * rgb * (1 - mask)
    return rgb


def lab2rgb(img_lab):
    # img_lab: torch.tensor().
    out = xyz2rgb(lab2xyz(img_lab))
    return out

