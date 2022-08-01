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

    fid_score, fid_score_convert = calculate_fid(pred_dir, gt_dir)
    print('fid_score', fid_score, 'fid_score_convert:', fid_score_convert)