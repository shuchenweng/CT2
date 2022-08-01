import os
from PIL import Image
import numpy as np
from skimage import color
import torch
import pickle
from pathlib import Path
import yaml
import json
import numpy as np
import torch
import click
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import torch.distributed
import warnings
import io

from segm.utils import distributed
import segm.utils.torch as ptu
from segm import config

from segm.model.factory import create_segmenter
from segm.optim.factory import create_optimizer, create_scheduler
from segm.data.factory import create_dataset
from segm.model.utils import num_params

from timm.utils import NativeScaler
from contextlib import suppress
import os

from segm.utils.distributed import sync_model
from segm.engine import train_one_epoch, evaluate
import collections
import skimage
# from segm.engine import save_imgs


def rgb_to_lab(img):
    assert img.dtype == np.uint8
    return color.rgb2lab(img).astype(np.float32)

def numpy_to_torch(img):
    tensor = torch.from_numpy(np.moveaxis(img, -1, 0))      # [c, h, w]
    return tensor.type(torch.float32)


def load_mask(mask_l_num):
    fp = open('/userhome/SUN_text2img/ImageNet/mask_prior.pickle', 'rb')
    L_dict = pickle.load(fp)
    mask_L = np.zeros((mask_l_num, 313)).astype(np.bool)
    for key in range(101):
        for ii in range(mask_l_num):
            start_key = ii * (100//mask_l_num)
            end_key = (ii + 1) * (100 // mask_l_num)
            if start_key <= key < end_key:
                mask_L[ii, :] += L_dict[key].astype(np.bool)
                break
    mask_L = mask_L.astype(np.float32)
    return mask_L


@click.command(help="")
@click.option("--log-dir", type=str, help="logging directory")
@click.option("--dataset", default='coco', type=str)
@click.option('--dataset_dir', default='/userhome/sjm/ImageNet',type=str)
@click.option("--im-size", default=256, type=int, help="dataset resize size")
@click.option("--crop-size", default=256, type=int)
@click.option("--window-size", default=256, type=int)
@click.option("--window-stride", default=None, type=int)
@click.option("--backbone", default="vit_tiny_patch16_384", type=str)       # try this, and freeze first several blocks.
@click.option("--decoder", default="mask_transformer", type=str)
@click.option("--optimizer", default="sgd", type=str)
@click.option("--scheduler", default="polynomial", type=str)
@click.option("--weight-decay", default=0.0, type=float)
@click.option("--dropout", default=0.0, type=float)
@click.option("--drop-path", default=0.1, type=float)
@click.option("--batch-size", default=None, type=int)
@click.option("--epochs", default=None, type=int)
@click.option("-lr", "--learning-rate", default=None, type=float)
@click.option("--normalization", default=None, type=str)
@click.option("--eval-freq", default=None, type=int)
@click.option("--amp/--no-amp", default=False, is_flag=True)
@click.option("--resume/--no-resume", default=True, is_flag=True)
@click.option('--local_rank', type=int)
@click.option('--only_test', type=bool, default=True)
@click.option('--add_mask', type=bool, default=True)        # valid
@click.option('--partial_finetune', type=bool, default=False)       # compare validation, last finetune all blocks.
@click.option('--add_l1_loss', type=bool, default=True)            # add after classification.
@click.option('--l1_weight', type=float, default=10)
@click.option('--color_position', type=bool, default=False)     # add color position in color token.
@click.option('--change_mask', type=bool, default=False)        # change mask, omit the attention between color tokens.
@click.option('--color_as_condition', type=bool, default=False)     # use self-attn to embedding color tokens, and use color to represent patch tokens.
@click.option('--multi_scaled', type=bool, default=False)       # multi-scaled decoder.
@click.option('--downchannel', type=bool, default=False)        # multi-scaled, upsample+downchannel. (should be correct??)
@click.option('--add_conv', type=bool, default=True)       # add conv after transformer blocks.
@click.option('--before_classify', type=bool, default=False)        # classification at 16x16 resolution, and use CNN upsampler for 256x256, then use l1-loss.
@click.option('--l1_conv', type=bool, default=True)                # patch--upsample--> [B, 256x256, 180]--conv3x3-> [B, 256x256, 2]
@click.option('--l1_linear', type=bool, default=False)          # pacth: [B, 16x16, 180]---linear-> [B, 16x16, 2x16x16]
@click.option('--add_fm', type=bool, default=False)             # add Feature matching loss.
@click.option('--fm_weight', type=float, default=1)
@click.option('--add_edge', type=bool, default=False)       # add sobel-conv to extract edge.
@click.option('--edge_loss_weight', type=float, default=0.05)     # edge_loss_weight.
@click.option('--mask_l_num', type=int, default=4)          # mask for L ranges: 4, 10, 20, 50, 100
@click.option('--n_blocks', type=int, default=1)        # per block have 2 layers. block_num = 2
@click.option('--without_colorattn', type=bool, default=False)
@click.option('--without_colorquery', type=bool, default=False)
@click.option('--without_classification', type=bool, default=False)
def test_func(
    log_dir,
    dataset,
    dataset_dir,
    im_size,
    crop_size,
    window_size,
    window_stride,
    backbone,
    decoder,
    optimizer,
    scheduler,
    weight_decay,
    dropout,
    drop_path,
    batch_size,
    epochs,
    learning_rate,
    normalization,
    eval_freq,
    amp,
    resume,
    local_rank,
    only_test,
    add_mask,
    partial_finetune,
    add_l1_loss,
    l1_weight,
    color_position,
    change_mask,
    color_as_condition,
    multi_scaled,
    downchannel,
    add_conv,
    before_classify,
    l1_conv,
    l1_linear,
    add_fm,
    fm_weight,
    add_edge,
    edge_loss_weight,
    mask_l_num,
    n_blocks,
    without_colorattn,
    without_colorquery,
    without_classification,
):
    # start distributed mode
    ptu.set_gpu_mode(True, local_rank)
    # distributed.init_process()
    torch.distributed.init_process_group(backend="nccl")

    # set up configuration
    cfg = config.load_config()
    model_cfg = cfg["model"][backbone]
    dataset_cfg = cfg["dataset"][dataset]
    if "mask_transformer" in decoder:
        decoder_cfg = cfg["decoder"]["mask_transformer"]
    else:
        decoder_cfg = cfg["decoder"][decoder]

    # model config
    if not im_size:
        im_size = dataset_cfg["im_size"]        # 256
    if not crop_size:
        crop_size = dataset_cfg.get("crop_size", im_size)       # 256
    if not window_size:
        window_size = dataset_cfg.get("window_size", im_size)
    if not window_stride:
        window_stride = dataset_cfg.get("window_stride", im_size)
    if not dataset_dir:
        dataset_dir = dataset_cfg.get('dataset_dir', None)

    model_cfg["image_size"] = (crop_size, crop_size)
    model_cfg["backbone"] = backbone
    model_cfg["dropout"] = dropout
    model_cfg["drop_path_rate"] = drop_path
    decoder_cfg["name"] = decoder
    model_cfg["decoder"] = decoder_cfg

    # dataset config
    world_batch_size = dataset_cfg["batch_size"]
    num_epochs = dataset_cfg["epochs"]
    lr = dataset_cfg["learning_rate"]

    if batch_size:
        world_batch_size = batch_size
    if epochs:
        num_epochs = epochs
    if learning_rate:
        lr = learning_rate
    if eval_freq is None:
        eval_freq = dataset_cfg.get("eval_freq", 1)

    if normalization:
        model_cfg["normalization"] = normalization

    # experiment config
    batch_size = world_batch_size // ptu.world_size
    variant = dict(
        world_batch_size=world_batch_size,
        version="normal",
        resume=resume,
        dataset_kwargs=dict(
            dataset=dataset,
            image_size=im_size,
            crop_size=crop_size,
            batch_size=batch_size,
            normalization=model_cfg["normalization"],
            split="train",
            num_workers=10,
            dataset_dir=dataset_dir,
            add_mask=add_mask,
            patch_size=model_cfg["patch_size"],
            change_mask=change_mask,
            multi_scaled=multi_scaled,
            mask_num=mask_l_num,
        ),
        algorithm_kwargs=dict(
            batch_size=batch_size,
            start_epoch=0,
            num_epochs=num_epochs,
            eval_freq=eval_freq,
        ),
        optimizer_kwargs=dict(
            opt=optimizer,
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.9,
            # momentum=0.9,
            clip_grad=None,
            sched=scheduler,
            epochs=num_epochs,
            min_lr=1e-5,
            poly_power=0.9,
            poly_step_size=1,
        ),
        net_kwargs=model_cfg,
        amp=amp,
        log_dir=log_dir,
        inference_kwargs=dict(
            im_size=im_size,
            window_size=window_size,
            window_stride=window_stride,
        ),
    )

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = log_dir / 'checkpoint_epoch_0_psnrcls_22.8164_psnrreg_24.5049.pth'

    # model
    net_kwargs = variant["net_kwargs"]
    net_kwargs["n_cls"] = 313

    net_kwargs['partial_finetune'] = partial_finetune
    net_kwargs['decoder']['add_l1_loss'] = add_l1_loss
    net_kwargs['decoder']['color_position'] = color_position
    net_kwargs['decoder']['change_mask'] = change_mask
    net_kwargs['decoder']['color_as_condition'] = color_as_condition
    net_kwargs['decoder']['multi_scaled'] = multi_scaled
    net_kwargs['decoder']['crop_size'] = crop_size
    net_kwargs['decoder']['downchannel'] = downchannel
    net_kwargs['decoder']['add_conv'] = add_conv
    net_kwargs['decoder']['before_classify'] = before_classify
    net_kwargs['decoder']['l1_conv'] = l1_conv
    net_kwargs['decoder']['l1_linear'] = l1_linear
    net_kwargs['decoder']['add_edge'] = add_edge
    net_kwargs['decoder']['n_blocks'] = n_blocks
    net_kwargs['decoder']['without_colorattn'] = without_colorattn
    net_kwargs['decoder']['without_colorquery'] = without_colorquery
    net_kwargs['decoder']['without_classification'] = without_classification
    model = create_segmenter(net_kwargs)
    model.to(ptu.device)

    amp_autocast = suppress
    loss_scaler = None
    if amp:
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()

    # resume
    if resume and checkpoint_path.exists():
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])

    if ptu.distributed:
        print('Distributed:', ptu.distributed)
        model = DDP(model, device_ids=[ptu.device], find_unused_parameters=True)

    # save config
    variant_str = yaml.dump(variant)
    print(f"Configuration:\n{variant_str}")
    variant["net_kwargs"] = net_kwargs
    # variant["dataset_kwargs"] = dataset_kwargs
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "variant.yml", "w") as f:
        f.write(variant_str)

    model_without_ddp = model
    if hasattr(model, "module"):
        model_without_ddp = model.module

    print(f"Encoder parameters: {num_params(model_without_ddp.encoder)}")
    print(f"Decoder parameters: {num_params(model_without_ddp.decoder)}")

    # load imgs.
    img_path = 'example.JPEG'

    img = Image.open(img_path).convert("RGB")
    img = np.array(img)
    l_resized = rgb_to_lab(img)[:, :, :1]
    ab_resized = rgb_to_lab(img)[:, :, 1:]  # np.float32
    original_l = l_resized[:, :, 0]
    l = original_l.reshape((256 * 256))

    mask_L = load_mask(mask_l_num)

    mask_p_c = np.zeros((256 ** 2, 313), dtype=np.float32)  # [256x256, 313]
    for l_range in range(mask_l_num):
        start_l1, end_l1 = l_range * (100//mask_l_num), (l_range+1) *(100//mask_l_num)
        if end_l1 == 100:
            index_l1 = np.where((l >= start_l1) & (l <= end_l1))[0]
        else:
            index_l1 = np.where((l >= start_l1) & (l < end_l1))[0]
        mask_p_c[index_l1, :] = mask_L[l_range, :]

    mask = torch.from_numpy(mask_p_c)  # [256*256, 313]

    img_l = numpy_to_torch(l_resized)
    img_ab = numpy_to_torch(ab_resized)
    mask, img_l, img_ab = mask.unsqueeze(0), img_l.unsqueeze(0), img_ab.unsqueeze(0)

    model_without_ddp = model
    if hasattr(model, "module"):
        model_without_ddp = model.module

    with torch.no_grad():
        img_l = img_l.to(ptu.device)
        img_ab = img_ab.to(ptu.device)
        mask = mask.to(ptu.device)
        ab_pred, q_pred, q_actual, out_feature = model_without_ddp.inference(img_l, img_ab, mask, appli=True)
        save_imgs(img_l, img_ab, ab_pred, "colorized.JPEG",
                  'saved_dir')



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

        img_path = os.path.join(dir, filenames)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            skimage.io.imsave(img_path, fake_rgb_list[j].astype(np.uint8))
            print('successful save imgs. ')

if __name__ == '__main__':
    test_func()







