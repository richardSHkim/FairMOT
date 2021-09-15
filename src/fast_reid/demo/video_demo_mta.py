# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import argparse
import logging
import sys
import os
import shutil

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch.backends import cudnn
import cv2

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
        "--dataset-name",
        help="a test dataset name for visualizing ranking list."
    )
    parser.add_argument(
        "--output",
        default="./vis_rank_list",
        help="a file or directory to save rankling list result.",
    )
    parser.add_argument(
        "--outdir",
        default="mta_demo",
        help="output directory"
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
        "--rank",
        default=3, type=int,
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

    test_loader, num_query = build_reid_test_loader(cfg, args.dataset_name)
    demo = FeatureExtractionDemo(cfg, parallel=args.parallel)

    logger.info("Start extracting image features")
    feats = []
    pids = []
    camids = []
    imgpaths = []
    for (feat, pid, camid, imgpath) in tqdm.tqdm(demo.run_on_vid_loader(test_loader), total=len(test_loader)):
        feats.append(feat)
        pids.extend(pid)
        camids.extend(camid)
        imgpaths.extend(imgpath)

    feats = torch.cat(feats, dim=0)
    q_feat = feats[:num_query]
    q_pids = np.asarray(pids[:num_query])
    q_camids = np.asarray(camids[:num_query])

    all_g_feat = feats[num_query:]
    all_g_pids = np.asarray(pids[num_query:])
    all_g_camids = np.asarray(camids[num_query:])
    all_g_imgpaths = np.asarray(imgpaths[num_query:])
    all_g_imgpaths = [x.split('/')[-1] for x in all_g_imgpaths]

    # tracklet
    re_id_frequency = 50
    tracklets = {}
    for i, feat, pid, camid, imgpath in zip(range(len(all_g_pids)), all_g_feat, all_g_pids, all_g_camids, all_g_imgpaths):
        # assert False, imgpath
        t = int(imgpath.split('_')[1])
        if not (t in tracklets.keys()):
            tracklet_dict = {}
            tracklet_dict['feat'] = []
            tracklet_dict['pids'] = []
            tracklet_dict['camids'] = []
            tracklet_dict['indices'] = []
            tracklets[t] = tracklet_dict
        tracklets[t]['feat'].append(feat)
        tracklets[t]['pids'].append(pid)
        tracklets[t]['camids'].append(camid)
        tracklets[t]['indices'].append(i)

    # compute cosine distance
    # distmat = 1 - torch.mm(q_feat_norm, g_feat_norm.t())

    for q_idx in range(0, 30):
        if os.path.isdir(args.output):
            shutil.rmtree(args.output)
        os.makedirs(args.output)
            
        for t in range(0, 4950, re_id_frequency):
            g_feat = torch.stack(tracklets[t]['feat'])
            g_pids = np.asarray(tracklets[t]['pids'])
            g_camids = np.asarray(tracklets[t]['camids'])
            g_indices = np.asarray(tracklets[t]['indices'])
            # query_features_norm = F.normalize(q_feat, p=2, dim=1)
            # gallery_features_norm = F.normalize(g_feat, p=2, dim=1)
            # score_norm = torch.mm(query_features_norm, gallery_features_norm.t()).cpu().numpy()
            score = torch.mm(q_feat, g_feat.t()).cpu().numpy()
            distmat = 1 - score

            if cfg.TEST.RERANK.ENABLED:
                k1 = cfg.TEST.RERANK.K1
                k2 = cfg.TEST.RERANK.K2
                lambda_value = cfg.TEST.RERANK.LAMBDA

                if cfg.TEST.METRIC == "cosine":
                    q_feat = F.normalize(q_feat, dim=1)
                    g_feat = F.normalize(g_feat, dim=1)

                rerank_dist = build_dist(q_feat, g_feat, metric="jaccard", k1=k1, k2=k2)
                distmat = rerank_dist * (1 - lambda_value) + distmat * lambda_value

            # logger.info("Computing APs for all query images ...")
            # cmc, all_ap, all_inp = evaluate_rank(distmat, q_pids, g_pids, q_camids, g_camids)
            # logger.info("Finish computing APs for all query images!")

            visualizer = Visualizer(test_loader.dataset) 
            visualizer.get_model_output(None, distmat, q_pids, g_pids, q_camids, g_camids)

            # logger.info("Start saving ROC curve ...")
            # fpr, tpr, pos, neg = visualizer.vis_roc_curve(args.output)
            # visualizer.save_roc_info(args.output, fpr, tpr, pos, neg)
            # logger.info("Finish saving ROC curve!")

            logger.info("Saving rank list result ... ID: %d / Tracklet: %d" % (q_idx, t))
            query_indices = visualizer.vis_rank_list_video_mta(q_idx, t, 0.4, args.output, args.vis_label, args.num_vis,
                                                    args.rank_sort, args.label_sort, max_cam=6, g_indices=g_indices, rank=args.rank, re_id_frequency=re_id_frequency)
            
            logger.info("Finish saving rank list results! ID: %d / Tracklet: %d" % (q_idx, t))
        
        files = [f for f in os.listdir(args.output)]
        if len(files)==0:
            continue
        files.sort()

        h, w, _ = cv2.imread(os.path.join(args.output, files[0])).shape
        vidout = cv2.VideoWriter(os.path.join(args.outdir, 'mta_out_query_%02d_rank_%d.avi' % (q_idx, args.rank)), cv2.VideoWriter_fourcc(*'DIVX'), 41, (w, h))

        for f in files:
            vidout.write(cv2.imread(os.path.join(args.output, f)))
        vidout.release()
        # shutil.rmtree(args.output)
        