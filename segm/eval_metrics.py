import torch
import argparse
from metrics import *
import numpy as np
import os


def parse_args():
    parser = argparse.ArgumentParser(description='evaluate')
    parser.add_argument('--pred_dir', help='colorized images')
    parser.add_argument('--gt_dir', help='groundtruth images')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    pred_dir = args.pred_dir
    gt_dir = args.gt_dir

    lpips, lpips_convert = avg_lpips(pred_dir, gt_dir)
    ssim, psnr, ssim_convert, psnr_convert = avg_ssim_psnr(pred_dir, gt_dir)

    print('ssim:', ssim, "ssim_convert:", ssim_convert, 'psnr:', psnr, 'psnr_convert:', psnr_convert, 'lpips:', lpips, 'lpips_convert:', lpips_convert)






