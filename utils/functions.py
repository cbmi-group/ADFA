from sklearn.metrics import roc_auc_score
import numpy as np
from scipy.ndimage import gaussian_filter
import torch.nn.functional as F

def rescale(x):
    return (x - x.min()) / (x.max() - x.min())

def gaussian_smooth(x, sigma=4):
    bs = x.shape[0]
    for i in range(0, bs):
        x[i] = gaussian_filter(x[i], sigma=sigma)
    return x

def upsample(x, size, mode):
    return F.interpolate(x.unsqueeze(1), size=size, mode=mode, align_corners=False).squeeze().numpy()

def cal_img_roc(scores, gt_list):
    img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
    gt_list = np.asarray(gt_list)
    img_roc_auc = roc_auc_score(gt_list, img_scores)
    return img_roc_auc