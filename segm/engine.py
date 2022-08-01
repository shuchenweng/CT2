import torch
import math

from segm.utils.logger import MetricLogger
from segm.metrics import gather_data, compute_metrics, classify_acc
from segm.model import utils
from segm.data.utils import IGNORE_LABEL
import segm.utils.torch as ptu
import torch.nn as nn
from torch.nn.functional import log_softmax
import torch.nn.functional as F
import numpy as np
from torchvision.models.inception import inception_v3
from torchvision.transforms import transforms
import os
from PIL import Image
from skimage import color, io
import warnings
from torch.autograd import Variable
from segm.metrics import INCEPTION_V3_FID, INCEPTION_V3, get_activations, calculate_frechet_distance, calculate_activation_statistics


class CrossEntropyLoss2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, labels, add_mask=None):
        B, n_cls, H, W = outputs.shape
        reshape_out = outputs.permute(0, 2, 3, 1).contiguous().view(B*H*W, n_cls)
        reshape_label = labels.permute(0, 2, 3, 1).contiguous().view(B*H*W, n_cls)       # [-1, 313]
        after_softmax = F.softmax(reshape_out, dim=1)
        # mask = add_mask.view(-1, n_cls)
        mask = after_softmax.clone()
        after_softmax = after_softmax.masked_fill(mask == 0, 1)
        out_softmax = torch.log(after_softmax)

        norm = reshape_label.clone()
        norm = norm.masked_fill(reshape_label == 0, 1)
        log_norm = torch.log(norm)

        loss = -torch.sum((out_softmax - log_norm) * reshape_label) / (B*H*W)
        return loss


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = utils.VGG19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


def functional_conv2d(im, input_channel=2, output_channel=2):
    conv_op = nn.Conv2d(input_channel, output_channel, 3, padding=1, bias=False).to(im.device)
    sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')  #
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))       # suitable for ab two channels.
    sobel_kernel_tensor = torch.from_numpy(sobel_kernel).repeat(output_channel, input_channel, 1, 1)
    conv_op.weight.data = sobel_kernel_tensor.to(im.device)
    edge_detect = conv_op(Variable(im))
    return edge_detect


def train_one_epoch(
    model,
    data_loader,
    optimizer,
    lr_scheduler,
    epoch,
    amp_autocast,
    loss_scaler,
    add_mask,
    add_l1_loss,
    l1_weight,
    partial_finetune,
    l1_conv,
    l1_linear,
    add_edge,
    edge_loss_weight,
    without_classification,
    log_dir,
):
    if not without_classification:
        criterion = CrossEntropyLoss2d()
    if add_l1_loss:
        loss_fn_l1 = nn.SmoothL1Loss()
    logger = MetricLogger(delimiter="  ")
    header = f"Epoch: [{epoch}]"
    print_freq = 100

    model.train()
    if partial_finetune:
        if hasattr(model, "module"):
            for block_layer in range(6):
                model.module.encoder.blocks[block_layer].eval()
        else:
            for block_layer in range(6):
                model.encoder.blocks[block_layer].eval()

    data_loader.set_epoch(epoch)
    num_updates = epoch * len(data_loader)

    train_loss_total, train_l1_total = 0, 0
    for batch in logger.log_every(data_loader, print_freq, header):
        img_l, img_ab, key, img_mask = batch
        img_l = img_l.to(ptu.device)
        img_ab = img_ab.to(ptu.device)
        if add_mask: img_mask = img_mask.to(ptu.device)

        with amp_autocast():
            if add_mask:
                ab_pred, q_pred, q_actual, out_feature = model.forward(img_l, img_ab, img_mask)       # out_feature: [B, 2, 256, 256]
            else:
                ab_pred, q_pred, q_actual, out_feature = model.forward(img_l, img_ab, None)

            if not without_classification:      # default False.
                loss = criterion(q_pred, q_actual)
            else:
                loss = 0

            if add_l1_loss:
                if l1_conv:
                    norm_ab = img_ab / 110.     # [-1, 1]
                    loss_l1 = loss_fn_l1(norm_ab, out_feature)
                elif l1_linear:
                    norm_ab = img_ab / 110.
                    loss_l1 = loss_fn_l1(norm_ab, out_feature)
                loss += loss_l1 * l1_weight

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss,
                optimizer,
                parameters=model.parameters(),
            )
        else:
            loss.backward()
            optimizer.step()
        loss_value = loss.item()
        train_loss_total += loss_value
        if add_l1_loss:
            train_l1_total += (loss_l1 * l1_weight).item()
        else:
            train_l1_total = 0
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value), force=True)

        num_updates += 1
        if num_updates % 5000 == 0 and ptu.dist_rank == 0:
            model_without_ddp = model
            if hasattr(model, "module"):
                model_without_ddp = model.module
            snapshot = dict(
                model=model_without_ddp.state_dict(),
                optimizer=optimizer.state_dict(),
                n_cls=model_without_ddp.n_cls,
                lr_scheduler=lr_scheduler.state_dict(),
            )
            if loss_scaler is not None:
                snapshot["loss_scaler"] = loss_scaler.state_dict()
            snapshot["epoch"] = epoch
            save_path = os.path.join(log_dir, 'checkpoint_epoch_%d_iter_%d.pth' % (epoch, num_updates))
            torch.save(snapshot, save_path)
            print('save model into:', save_path)

        lr_scheduler.step_update(num_updates=num_updates)

        torch.cuda.synchronize()

    logger.update(
        loss=train_loss_total/len(data_loader),
        loss_l1=train_l1_total/len(data_loader),
        learning_rate=optimizer.param_groups[0]["lr"],
    )
    return logger


@torch.no_grad()
def evaluate(
    epoch,
    model,
    data_loader,
    window_size,
    window_stride,
    amp_autocast,
    add_mask,
    add_l1_loss,
    l1_weight,
    l1_conv,
    l1_linear,
    add_fm,
    fm_weight,
    log_dir=None,
    diversity_index=0
):
    model_without_ddp = model
    if hasattr(model, "module"):
        model_without_ddp = model.module
    logger = MetricLogger(delimiter="  ")
    header = "Eval:"
    print_freq = 50
    if log_dir is not None:
        save_dir = os.path.join(log_dir, 'color_token_nums_78')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
    model.eval()
    total_psnr_cls, total_psnr_reg, fid_score = 0, 0, 0
    with torch.no_grad():
        for batch in logger.log_every(data_loader, print_freq, header):
            img_l, img_ab, filename, img_mask = batch
            img_l = img_l.to(ptu.device)
            img_ab = img_ab.to(ptu.device)
            if add_mask: img_mask = img_mask.to(ptu.device)

            with amp_autocast():
                if add_mask:
                    ab_pred, q_pred, q_actual, out_feature = model_without_ddp.inference(img_l, img_ab, img_mask)
                else:
                    ab_pred, q_pred, q_actual, out_feature = model_without_ddp.inference(img_l, img_ab, None)

            if log_dir is not None:
                if ab_pred is not None:
                    save_imgs(img_l, img_ab, ab_pred, filename, save_dir)

    logger.update(eval_psnr_cls=total_psnr_cls/len(data_loader),
                  eval_psnr_reg=total_psnr_reg/len(data_loader),
                  eval_fid=fid_score)
    return logger


def lab_to_rgb(img):
    assert img.dtype == np.float32
    return (255 * np.clip(color.lab2rgb(img), 0, 1)).astype(np.uint8)


def save_imgs(img_l, img_ab, ab_pred, filenames, dir):
    img_lab = torch.cat((img_l, ab_pred.detach()), dim=1).cpu()
    batch_size = img_lab.size(0)
    fake_rgb_list, real_rgb_list, only_rgb_list = [], [], []
    for j in range(batch_size):
        img_lab_np = img_lab[j].numpy().transpose(1, 2, 0)      # np.float32
        img_rgb = lab_to_rgb(img_lab_np)        # np.uint8      # [0-255]
        fake_rgb_list.append(img_rgb)

        img_path = os.path.join(dir, 'fake_' + filenames[j])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            io.imsave(img_path, fake_rgb_list[j].astype(np.uint8))


def calculate_psnr(img_l, img_ab, ab_pred, out_feature):
    real_lab = torch.cat((img_l, img_ab), dim=1).cpu()
    if ab_pred is not None:
        fake_lab = torch.cat((img_l, ab_pred.detach()), dim=1).cpu()
    if out_feature is not None:
        fake_lab_reg = torch.cat((img_l, (out_feature * 110).detach()), dim=1).cpu()
    bs = real_lab.size(0)
    assert bs == 1
    ##############
    psnr_cls, psnr_reg = 0, 0
    for j in range(bs):
        real_lab_np = real_lab[j].numpy().transpose(1, 2, 0)
        real_rgb = lab_to_rgb(real_lab_np)
        if ab_pred is not None:
            fake_lab_np = fake_lab[j].numpy().transpose(1, 2, 0)
            fake_rgb = lab_to_rgb(fake_lab_np)
            each_psnr = calculate_psnr_np(fake_rgb, real_rgb)
            psnr_cls += each_psnr

        if out_feature is not None:
            fake_lab_reg = fake_lab_reg[j].numpy().transpose(1, 2, 0)
            fake_rgb_reg = lab_to_rgb(fake_lab_reg)
            each_psnr_reg = calculate_psnr_np(fake_rgb_reg, real_rgb)
            psnr_reg += each_psnr_reg
    psnr_cls = psnr_cls / bs
    psnr_reg = psnr_reg / bs

    return psnr_cls, psnr_reg


def calculate_psnr_np(img1, img2):
    import numpy as np
    SE_map = (1. * img1 - img2) ** 2
    cur_MSE = np.mean(SE_map)
    return 20 * np.log10(255. / np.sqrt(cur_MSE))


def lab2xyz(lab):
    y_int = (lab[:,0,:,:]+16.)/116.
    x_int = (lab[:,1,:,:]/500.) + y_int
    z_int = y_int - (lab[:,2,:,:]/200.)
    if(z_int.is_cuda):
        z_int = torch.max(torch.Tensor((0,)).cuda(), z_int)
    else:
        z_int = torch.max(torch.Tensor((0,)), z_int)

    out = torch.cat((x_int[:,None,:,:],y_int[:,None,:,:],z_int[:,None,:,:]),dim=1)
    mask = (out > .2068966).type(torch.FloatTensor)
    if(out.is_cuda):
        mask = mask.cuda()

    out = (out**3.)*mask + (out - 16./116.)/7.787*(1-mask)

    sc = torch.Tensor((0.95047, 1., 1.08883))[None,:,None,None]
    sc = sc.to(out.device)

    out = out*sc
    return out


def xyz2rgb(xyz):
    r = 3.24048134*xyz[:,0,:,:]-1.53715152*xyz[:,1,:,:]-0.49853633*xyz[:,2,:,:]
    g = -0.96925495*xyz[:,0,:,:]+1.87599*xyz[:,1,:,:]+.04155593*xyz[:,2,:,:]
    b = .05564664*xyz[:,0,:,:]-.20404134*xyz[:,1,:,:]+1.05731107*xyz[:,2,:,:]

    rgb = torch.cat((r[:,None,:,:],g[:,None,:,:],b[:,None,:,:]),dim=1)
    rgb = torch.max(rgb,torch.zeros_like(rgb)) # sometimes reaches a small negative number, which causes NaNs

    mask = (rgb > .0031308).type(torch.FloatTensor)
    if(rgb.is_cuda):
        mask = mask.cuda()

    rgb = (1.055*(rgb**(1./2.4)) - 0.055)*mask + 12.92*rgb*(1-mask)

    return rgb

