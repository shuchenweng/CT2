from imutils import build_montages
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os

def parse_args():
    parser = argparse.ArgumentParser(description='evaluate')
    parser.add_argument('--pred_dir', help='colorized images')
    args = parser.parse_args()
    return args

def image_colorfulness(image):
    (B, G, R) = cv2.split(image.astype('float'))
    rg = np.absolute(R - G)
    yb = np.absolute(0.5 * (R+G) - B)
    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))
    stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
    return stdRoot + (0.3 * meanRoot)

if __name__ == '__main__':
    args = parse_args()
    pred_dir = args.pred_dir
    filename_list = os.listdir(pred_dir)
    assert len(filename_list) == 5000
    total_colorfullness = 0
    ii = 0
    for img_pth in filename_list:
        print('ii', ii)
        image_pth = os.path.join(pred_dir, img_pth)
        image = cv2.imread(image_pth)
        C = image_colorfulness(image)
        total_colorfullness += C
        ii += 1
    colorfullness = total_colorfullness/len(filename_list)

