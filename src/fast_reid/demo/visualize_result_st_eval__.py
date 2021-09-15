# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import argparse
import logging
import sys

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch.backends import cudnn

sys.path.append('.')

from fastreid.evaluation import evaluate_rank
from fastreid.config import get_cfg
from fastreid.utils.logger import setup_logger
from fastreid.data import build_reid_st_test_loader
from predictor import FeatureExtractionDemo
from fastreid.utils.visualizer import Visualizer

from fastreid.utils.spatial_temporal import st_evaluation
from fastreid.utils.compute_dist import build_dist
from fastreid.evaluation.rerank import re_ranking2

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
        "--dataset-name",
        help="a test dataset name for visualizing ranking list."
    )
    parser.add_argument(
        "--output",
        default="./vis_rank_list",
        help="a file or directory to save rankling list result.",

    )
    parser.add_argument(
        "--vis-label",
        action='store_true',
        help="if visualize label of query instance"
    )
    parser.add_argument(
        "--num-vis", type=int,
        default=0,
        help="number of query images to be visualized",
    )
    parser.add_argument(
        "--rank-sort", type=str,
        default="descending",
        help="rank order of visualization images by AP metric",
    )
    parser.add_argument(
        "--label-sort", type=str,
        default="descending",
        help="label order of visualization images by cosine similarity metric",
    )
    parser.add_argument(
        "--max-rank", type=int,
        default=10,
        help="maximum number of rank list to be visualized",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    cfg = setup_cfg(args)
    test_loader, num_query = build_reid_st_test_loader(cfg, args.dataset_name)
    demo = FeatureExtractionDemo(cfg, parallel=args.parallel)

    logger.info("Start extracting image features")
    feats = []
    pids = []
    camids = []
    frames = []
    for (feat, pid, camid, frame) in tqdm.tqdm(demo.run_on_loader(test_loader), total=len(test_loader)):
        feats.append(feat)
        pids.extend(pid)
        camids.extend(camid)
        frames.extend(frame)

    feats = torch.cat(feats, dim=0)
    q_feat = feats[:num_query]
    g_feat = feats[num_query:]
    q_pids = np.asarray(pids[:num_query])
    g_pids = np.asarray(pids[num_query:])
    q_camids = np.asarray(camids[:num_query])
    g_camids = np.asarray(camids[num_query:])
    q_frames = np.asarray(frames[:num_query])
    g_frames = np.asarray(frames[num_query:])

    # compute cosine distance
    # distmat = 1 - torch.mm(q_feat_norm, g_feat_norm.t())
    query_features_norm = F.normalize(q_feat, p=2, dim=1)
    gallery_features_norm = F.normalize(g_feat, p=2, dim=1)
    score = torch.mm(query_features_norm, gallery_features_norm.t()).cpu().numpy()

    # SPATIAL TEMPORAL SCORE ####################################
    result2 = scipy.io.loadmat(os.path.join(cfg.OUTPUT_DIR, '..', 'pytorch_result2.mat'))
    distribution = result2['distribution']
    score = st_evaluation(query_features, query_pids, query_camids, query_frames, 
                gallery_features, gallery_pids, gallery_camids, gallery_frames,
                score, 
                alpha=self.cfg.TEST.SPATIALTEMPORAL.ALPHA, smooth=self.cfg.TEST.SPATIALTEMPORAL.SMOOTH, distribution=distribution)
    distmat = 1 - score
    #############################################################

    if cfg.TEST.RERANK.ENABLED:
        k1 = cfg.TEST.RERANK.K1
        k2 = cfg.TEST.RERANK.K2
        lambda_value = cfg.TEST.RERANK.LAMBDA

        features_norm = F.normalize(features, p=2, dim=1)
        score = torch.mm(features_norm, features_norm.t()).cpu().numpy()

        # features = features
        pids = np.asarray(pids)
        camids = np.asarray(camids)
        frames = np.asarray(frames)

        all_scores = np.zeros((len(pids), len(pids)))
        all_scores = st_evaluation(feats, pids, camids, frames,
                            feats, pids, camids, frames,
                            score,
                            alpha=cfg.TEST.SPATIALTEMPORAL.ALPHA, smooth=cfg.TEST.SPATIALTEMPORAL.SMOOTH, 
                            distribution=distribution)

        rerank_dist = re_ranking2(num_query, all_scores)

        distmat = rerank_dist * (1 - lambda_value) + distmat * lambda_value

    logger.info("Computing APs for all query images ...")
    cmc, all_ap, all_inp = evaluate_rank(distmat, q_pids, g_pids, q_camids, g_camids)
    logger.info("Finish computing APs for all query images!")

    visualizer = Visualizer(test_loader.dataset) 
    visualizer.get_model_output(all_ap, distmat, q_pids, g_pids, q_camids, g_camids)

    logger.info("Start saving ROC curve ...")
    fpr, tpr, pos, neg = visualizer.vis_roc_curve(args.output)
    visualizer.save_roc_info(args.output, fpr, tpr, pos, neg)
    logger.info("Finish saving ROC curve!")

    logger.info("Saving rank list result ...")
    query_indices = visualizer.vis_rank_list(args.output, args.vis_label, args.num_vis,
                                             args.rank_sort, args.label_sort, args.max_rank)
    logger.info("Finish saving rank list results!")
