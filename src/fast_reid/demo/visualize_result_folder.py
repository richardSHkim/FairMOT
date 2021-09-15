# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import argparse
import logging
import sys
import os

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch.backends import cudnn
import glob
import cv2
import matplotlib.pyplot as plt

sys.path.append('.')

from fastreid.evaluation import evaluate_rank
from fastreid.config import get_cfg
from fastreid.utils.logger import setup_logger
from fastreid.data import build_reid_test_loader
from predictor import FeatureExtractionDemo
from fastreid.utils.visualizer import Visualizer

from fastreid.utils.compute_dist import build_dist

# import some modules added in project
# for example, add partial reid like this below
# from projects.PartialReID.partialreid import *

cudnn.benchmark = True
setup_logger(name="fastreid")

logger = logging.getLogger('fastreid.visualize_result')


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Feature extraction with reid models")
    parser.add_argument(
        "--config-file",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='if use multiprocess for feature extraction.'
    )
    parser.add_argument(
        "--gallery-dir",
        default='demo/gallery'
    )
    parser.add_argument(
        "--query-dir",
        default='demo/query'
    )
    parser.add_argument(
        "--output",
        default="./folder_demo_results",
        help="a file or directory to save rankling list result.",

    )
    parser.add_argument(
        "--vis-label",
        action='store_true',
        help="if visualize label of query instance"
    )
    parser.add_argument(
        "--num-vis",
        default=0, type=int,
        help="number of query images to be visualized",
    )
    parser.add_argument(
        "--rank-sort",
        default="descending", type=str,
        help="rank order of visualization images by AP metric",
    )
    parser.add_argument(
        "--label-sort",
        default="descending", type=str,
        help="label order of visualization images by cosine similarity metric",
    )
    parser.add_argument(
        "--max-rank",
        default=10, type=int,
        help="maximum number of rank list to be visualized",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def vis_rank_list(q_imgs, g_imgs, distmat, output, max_rank=5):
    fig, axes = plt.subplots(1, max_rank + 1, figsize=(3 * max_rank, 6))
    indices = np.argsort(distmat, axis=1)

    for cnt, q_img in enumerate(q_imgs):
        plt.clf()
        ax = fig.add_subplot(1, max_rank + 1, 1)
        q_img = q_img[:, :, ::-1]
        ax.imshow(q_img)
        ax.set_title('qid_{}'.format(0))
        ax.axis("off")

        for i in range(max_rank):
            g_idx = indices[cnt, i]
            ax = fig.add_subplot(1, max_rank + 1, i + 2)
            g_img = g_imgs[g_idx]
            g_img = g_img[:, :, ::-1]
            ax.add_patch(plt.Rectangle(xy=(0, 0), width=g_img.shape[1] - 1,
                                        height=g_img.shape[0] - 1, edgecolor=(1, 1, 1), 
                                        fill=False, linewidth=5))
            ax.imshow(g_img)
            ax.set_title('%.2f' % (1-distmat[cnt, g_idx]))
            ax.axis("off")

        plt.tight_layout()
        filepath = os.path.join(output, "{}.jpg".format(cnt))
        fig.savefig(filepath)


if __name__ == '__main__':
    args = get_parser().parse_args()
    cfg = setup_cfg(args)

    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    
    demo = FeatureExtractionDemo(cfg, parallel=args.parallel)

    logger.info("Start extracting gallery image features")
    g_imgs = []
    g_feats = []
    for g_img_name in glob.glob(os.path.join(args.gallery_dir, '*.jpg')):
        g_img = cv2.imread(g_img_name)
        feat = demo.run_on_image(g_img)
        g_imgs.append(g_img)
        g_feats.append(feat)
    g_feat = torch.cat(g_feats, dim=0)

    logger.info("Start extracting query image features")
    q_imgs = []
    q_feats = []
    for q_img_name in glob.glob(os.path.join(args.query_dir, '*.jpg')):
        q_img = cv2.imread(q_img_name)
        feat = demo.run_on_image(q_img)
        q_imgs.append(q_img)
        q_feats.append(feat)
    q_feat = torch.cat(q_feats, dim=0)

    # compute cosine distance
    query_features_norm = F.normalize(q_feat, p=2, dim=1)
    gallery_features_norm = F.normalize(g_feat, p=2, dim=1)
    score = torch.mm(query_features_norm, gallery_features_norm.t()).cpu().numpy()
    distmat = 1 - score  # [num_q, num_g]

    if cfg.TEST.RERANK.ENABLED:
        k1 = cfg.TEST.RERANK.K1
        k2 = cfg.TEST.RERANK.K2
        lambda_value = cfg.TEST.RERANK.LAMBDA

        if cfg.TEST.METRIC == "cosine":
            q_feat = F.normalize(q_feat, dim=1)
            g_feat = F.normalize(g_feat, dim=1)

        rerank_dist = build_dist(q_feat, g_feat, metric="jaccard", k1=k1, k2=k2)
        distmat = rerank_dist * (1 - lambda_value) + distmat * lambda_value
    
    logger.info("Saving rank list result ...")
    query_indices = vis_rank_list(q_imgs=q_imgs, g_imgs=g_imgs, distmat=distmat, output=args.output, max_rank=args.max_rank)
    logger.info("Finish saving rank list results!")
