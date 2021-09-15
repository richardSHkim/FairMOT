from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import os.path as osp
import cv2
import logging
import argparse
import motmetrics as mm
import numpy as np
import torch
import torch.nn.functional as F

from tracker.multitracker import JDETracker
from tracker.multitracker_reid import JDETrackerReID
from tracker.multitracker_mtmct import JDETrackerMTMCT
from tracking_utils import visualization as vis
from tracking_utils.log import logger
from tracking_utils.timer import Timer
from tracking_utils.evaluation import Evaluator
import datasets.dataset.jde as datasets

from tracking_utils.utils import mkdir_if_missing
from opts import opts

# reid
from fast_reid.fastreid.config import get_cfg
from fast_reid.fastreid.utils.logger import setup_logger
from fast_reid.fastreid.utils.file_io import PathManager
from fast_reid.demo.predictor import FeatureExtractionDemo


def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def write_results_score(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},{s},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h, s=score)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def eval_seq(opt, dataloader, data_type, result_filename, save_dir=None, show_image=True, frame_rate=30, use_cuda=True, out=None, save_bbox=False):
    if save_dir:
        mkdir_if_missing(save_dir)
    tracker = JDETracker(opt, frame_rate=frame_rate)
    timer = Timer()
    results = []
    frame_id = 0
    bbox_count = 0
    #for path, img, img0 in dataloader:
    for i, (path, img, img0) in enumerate(dataloader):
        #if i % 8 != 0:
            #continue
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        # run tracking
        timer.tic()
        if use_cuda:
            blob = torch.from_numpy(img).cuda().unsqueeze(0)
        else:
            blob = torch.from_numpy(img).unsqueeze(0)
        online_targets = tracker.update(blob, img0)
        online_tlwhs = []
        online_ids = []
        #online_scores = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                #online_scores.append(t.score)
        timer.toc()
        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))
        #results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))
        if save_bbox and (frame_id % 20 == 0):
            for tlwh in online_tlwhs:
                tlwh = [int(x) for x in tlwh]
                x1, y1, w, h = tlwh
                bbox = img0[y1:y1+h, x1:x1+w]
                if (bbox.shape[0] < 10) or (bbox.shape[1] < 10):
                    continue
                cv2.imwrite(os.path.join(save_dir, '%d_%d.jpg' % (frame_id, bbox_count)), bbox)
                bbox_count += 1

        if show_image or save_dir is not None:
            online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
                                          fps=1. / timer.average_time)
        if show_image:
            cv2.imshow('online_im', online_im)
        # if save_dir is not None:
        #     cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
        if out != None:
            out.write(image=online_im)
        frame_id += 1
    # save results
    write_results(result_filename, results, data_type)
    #write_results_score(result_filename, results, data_type)
    return frame_id, timer.average_time, timer.calls


def eval_seq_multicam(opt, dataloaders, data_type, result_filename, save_dir=None, show_image=True, frame_rate=30, use_cuda=True):
    if save_dir:
        mkdir_if_missing(save_dir)
    tracker = JDETrackerReID(opt, frame_rate=frame_rate)
    timer = Timer()
    results = []
    frame_id = 0

    vids_len = min([dloader.__len__() for dloader in dataloaders])
    online_cids = 0
    q_cid = None
    query = cv2.imread(opt.query)
    reid_iter = 20
    for i in range(vids_len-1):
        imgs = {}
        imgs0 = {}

        for c, dloader in enumerate(dataloaders):
            _, img, img0 = dloader.__next__()
            if (i%reid_iter)==0:
                imgs[c] = img
                imgs0[c] = img0
            elif q_cid==c:
                imgs[c] = img
                imgs0[c] = img0
            else:
                continue
        
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        # run tracking
        timer.tic()
        blobs = {}
        if use_cuda:
            for c, img in imgs.items():
                blobs[c] = torch.from_numpy(img).cuda().unsqueeze(0)
        else:
            for c, img in imgs.items():
                blobs[c] = torch.from_numpy(img).unsqueeze(0)

        online_targets = tracker.update_multicam(blobs, imgs0, query)
        online_tlwhs = []
        online_ids = []
        #online_scores = []

        if (i%reid_iter)==0:
            q_id, q_cid, q_score = find_query(reid_demo, imgs0, online_targets, opt.query, thresh=0.4)

        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            if tid != q_id:
                continue
            # tcid = t.track_camid
            # x1, y1, w, h = tlwh
            # intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
            # assert False, (tlwh, tid)
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_cids= t.cid
                #online_scores.append(t.score)
        timer.toc()
            
        # if len(online_tlwhs)==0 or (q_id==None):
        #     q_id, q_cid, q_score = find_query(reid_demo, imgs0, online_targets, opt.query)

        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))
        #results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))

        if show_image or save_dir is not None:
            online_im = vis.plot_tracking_multicam(imgs0[online_cids], online_tlwhs, online_ids, frame_id=frame_id,
                                          fps=1. / timer.average_time, cid=online_cids, score=q_score)
        if show_image:
            cv2.imshow('online_im', online_im)
        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
        frame_id += 1
    # save results
    write_results(result_filename, results, data_type)
    #write_results_score(result_filename, results, data_type)
    return frame_id, timer.average_time, timer.calls


def eval_seq_query(opt, dataloader, data_type, result_filename, save_dir=None, show_image=True, frame_rate=30, use_cuda=True, out=None):
    if save_dir:
        mkdir_if_missing(save_dir)
    
    # tracking
    tracker = JDETracker(opt, frame_rate=frame_rate)

    # re-id
    print('Creating Re-id model')
    cfg = setup_cfg_reid(opt)
    reid_demo = FeatureExtractionDemo(cfg, parallel=False)

    timer = Timer()
    results = []
    frame_id = 0

    query = cv2.imread(opt.query)
    reid_iter = 20

    for i, (path, img, img0) in enumerate(dataloader):
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        # run tracking
        timer.tic()
        if use_cuda:
            blob = torch.from_numpy(img).cuda().unsqueeze(0)
        else:
            blob = torch.from_numpy(img).unsqueeze(0)

        online_targets = tracker.update(blob, img0)
        online_tlwhs = []
        online_ids = []
        online_scores = []

        # if (i%reid_iter)==0:
        #     query_track, q_score = find_query(reid_demo, img0, online_targets, query, thresh=0.0)

        if ((i%reid_iter) == 0) and (len(online_targets) != 0):
            q_track = get_reid_score(reid_demo, img0, online_targets, query)
        
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            if q_track is None:
                continue
            if tid != q_track.track_id:
                continue
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(t.reid_score)
        timer.toc()
            
        # if len(online_tlwhs)==0 or (q_id==None):
        #     q_id, q_cid, q_score = find_query(reid_demo, imgs0, online_targets, opt.query)

        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))
        #results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))

        if show_image or save_dir is not None:
            online_im = vis.plot_tracking_query(img0, online_tlwhs, online_ids, frame_id=frame_id,
                                          fps=1. / timer.average_time, scores=online_scores)
        if show_image:
            cv2.imshow('online_im', online_im)
        # if save_dir is not None:
        #     cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
        if out != None:
            out.write(image=online_im)
        frame_id += 1
    # save results
    write_results(result_filename, results, data_type)
    #write_results_score(result_filename, results, data_type)
    return frame_id, timer.average_time, timer.calls


def eval_seq_mtmct(opt, dataloaders, data_type, result_filename, save_dir=None, show_image=True, frame_rate=30, use_cuda=True, 
                    out=None, vw=None, vh=None):
    if save_dir:
        mkdir_if_missing(save_dir)
    
    # tracking
    tracker = JDETrackerMTMCT(opt, frame_rate=frame_rate)

    # re-id
    print('Creating Re-id model')
    cfg = setup_cfg_reid(opt)
    reid_demo = FeatureExtractionDemo(cfg, parallel=False)

    timer = Timer()
    results = []
    frame_id = 0

    vids_len = min([dloader.__len__() for c, dloader in dataloaders.items()])
    vis_cid = list(dataloaders.keys())[0]
    query = cv2.imread(opt.query)

    # for vis - query im
    qw = vw - int(vw/11)*10
    qh = int(query.shape[0] * (qw / query.shape[1]))
    query_for_vis = cv2.resize(query, (qw, qh))

    qtw = int(vw/11)
    q_track_empty = np.ones([qh, qtw, 3])*255
    q_track_list = [q_track_empty] * 10

    query_im = np.concatenate([query_for_vis] + q_track_list, axis=1)

    # for vis - query re-id score
    sh = int(qh/10)
    reid_score_emtpy = np.ones([sh, qtw, 3])*255
    query_title = cv2.putText(np.ones([sh, qw, 3])*255, 'query', 
                            (0, int(sh/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), thickness=2)
    reid_score_vis_list = [reid_score_emtpy] * 10
    score_im = np.concatenate([query_title] + reid_score_vis_list, axis=1)
    reid_score_list = [0]*10

    reid_iter = 80
    for i in range(vids_len-1):
        imgs = {}
        imgs0 = {}

        for c, dloader in dataloaders.items():
            _, img, img0 = dloader.__next__()
            imgs[c] = img
            imgs0[c] = img0
            # if (i%reid_iter)==0:
            #     imgs[c] = img
            #     imgs0[c] = img0
            # elif c==vis_cid:
            #     imgs[c] = img
            #     imgs0[c] = img0
            # else:
            #     continue

        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        # run tracking
        timer.tic()
        blobs = {}
        if use_cuda:
            for c, img in imgs.items():
                blobs[c] = torch.from_numpy(img).cuda().unsqueeze(0)
        else:
            for c, img in imgs.items():
                blobs[c] = torch.from_numpy(img).unsqueeze(0)

        online_targets = tracker.update(blobs, imgs0)
        online_tlwhs = []
        online_ids = []
        online_scores = []

        if ((i%reid_iter) == 0) and (len(online_targets) != 0):
            q_track, q_track_img = get_reid_score_mtmct(reid_demo, imgs0, online_targets, query)
            if q_track != None:
                vis_cid = q_track.cid

        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            if q_track is None:
                continue
            if tid != q_track.track_id:
                continue
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(t.reid_score)
                # vis_cid = t.cid
        timer.toc()

        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))

        if show_image or save_dir is not None:
            online_im = vis.plot_tracking_mtmct(imgs0[vis_cid], online_tlwhs, online_ids, frame_id=frame_id,
                                          fps=1. / timer.average_time, cid=vis_cid, scores=online_scores)
            if ((i%reid_iter) == 0) and (len(online_targets) != 0) and (q_track != None):
                q_track_img = cv2.resize(q_track_img, dsize=(qtw, qh))
                reid_score_vis_img = cv2.putText(np.copy(reid_score_emtpy), '%.2f' % online_scores[0], 
                        (0, int(sh/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), thickness=2)

                q_track_list.insert(0, q_track_img)
                reid_score_vis_list.insert(0, reid_score_vis_img)
                reid_score_list.insert(0, online_scores[0])

                q_track_list = [x for _, x in sorted(zip(reid_score_list, q_track_list), reverse=True)]
                reid_score_vis_list = [x for _, x in sorted(zip(reid_score_list, reid_score_vis_list), reverse=True)]
                reid_score_list.sort(reverse=True)

                q_track_list.pop(-1)
                reid_score_vis_list.pop(-1)
                reid_score_list.pop(-1)

                query_im = np.concatenate([query_for_vis] + q_track_list, axis=1)
                score_im = np.concatenate([query_title] + reid_score_vis_list, axis=1)

            online_im = np.concatenate([online_im, query_im], axis=0)
            online_im = np.concatenate([online_im, score_im], axis=0)
            
        if out != None:
            online_im=np.uint8(online_im)
            # cv2.imwrite('tmp/%05d.jpg' % frame_id, online_im)
            out.write(image=online_im)
        frame_id += 1
    # save results
    write_results(result_filename, results, data_type)
    #write_results_score(result_filename, results, data_type)
    return frame_id, timer.average_time, timer.calls


def find_query(reid_demo, imgs0, online_targets, q_img, thresh):
    q_feat = []
    feat = reid_demo.run_on_image(q_img)
    q_feat.append(feat)
    q_feat = torch.cat(q_feat, dim=0)

    g_feat = []
    for t in online_targets:
        x1, y1, w, h = t.tlwh
        bbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        img0 = imgs0[t.cid]
        g_img = img0[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
        if (g_img.shape[0] < 10) or (g_img.shape[1] < 10):
            continue
        feat = reid_demo.run_on_image(g_img)
        g_feat.append(feat)
    if len(g_feat)==0:
        return None, None
    g_feat = torch.cat(g_feat, dim=0)

    q_feat = F.normalize(q_feat, p=2, dim=1)
    g_feat = F.normalize(g_feat, p=2, dim=1)
    dist = (1 - torch.mm(q_feat, g_feat.t())).numpy()
    indices = np.argsort(dist, axis=1)[0]

    query_track = online_targets[indices[0]]
    q_score = 1 - dist[0, indices[0]]
    # print(q_score)
    # if q_score < thresh:
    #     return None, None, q_score
    # assert False
    return query_track, q_score


def get_reid_score(reid_demo, img0, online_targets, q_img):
    q_feat = []
    feat = reid_demo.run_on_image(q_img)
    q_feat.append(feat)
    q_feat = torch.cat(q_feat, dim=0)

    g_feat = []
    for t in online_targets:
        x1, y1, w, h = t.tlwh
        bbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        g_img = img0[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
        if (g_img.shape[0] < 10) or (g_img.shape[1] < 10):
            continue
        feat = reid_demo.run_on_image(g_img)
        g_feat.append(feat)
    if len(g_feat)==0:
        return None
    g_feat = torch.cat(g_feat, dim=0)

    q_feat = F.normalize(q_feat, p=2, dim=1)
    g_feat = F.normalize(g_feat, p=2, dim=1)
    online_scores = torch.mm(q_feat, g_feat.t()).numpy()

    dist = (1 - torch.mm(q_feat, g_feat.t())).numpy()
    indices = np.argsort(dist, axis=1)[0]

    for idx in indices:
        online_targets[idx].reid_score = online_scores[0, idx]

    q_track = online_targets[indices[0]]
    return q_track


def get_reid_score_mtmct(reid_demo, imgs0, online_targets, q_img):
    q_feat = []
    feat = reid_demo.run_on_image(q_img)
    q_feat.append(feat)
    q_feat = torch.cat(q_feat, dim=0)

    g_feat = []
    g_imgs = []
    for t in online_targets:
        x1, y1, w, h = t.tlwh
        bbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        g_img = imgs0[t.cid][bbox[1]:bbox[3], bbox[0]:bbox[2], :]
        if (g_img.shape[0] < 10) or (g_img.shape[1] < 10):
            continue
        feat = reid_demo.run_on_image(g_img)
        g_feat.append(feat)
        g_imgs.append(g_img)
    if len(g_feat)==0:
        return None, None
    g_feat = torch.cat(g_feat, dim=0)

    q_feat = F.normalize(q_feat, p=2, dim=1)
    g_feat = F.normalize(g_feat, p=2, dim=1)
    online_scores = torch.mm(q_feat, g_feat.t()).numpy()

    dist = (1 - torch.mm(q_feat, g_feat.t())).numpy()
    indices = np.argsort(dist, axis=1)[0]

    for idx in indices:
        online_targets[idx].reid_score = online_scores[0, idx]

    q_track = online_targets[indices[0]]

    if q_track.reid_score < 0.3:
        return None, None
    return q_track, g_imgs[indices[0]]


def setup_cfg_reid(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # add_partialreid_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def main(opt, data_root='/data/MOT16/train', det_root=None, seqs=('MOT16-05',), exp_name='demo',
         save_images=False, save_videos=False, show_image=True):
    logger.setLevel(logging.INFO)
    result_root = os.path.join(data_root, '..', 'results', exp_name)
    mkdir_if_missing(result_root)
    data_type = 'mot'

    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []
    for seq in seqs:
        output_dir = os.path.join(data_root, '..', 'outputs', exp_name, seq) if save_images or save_videos else None
        logger.info('start seq: {}'.format(seq))
        dataloader = datasets.LoadImages(osp.join(data_root, seq, 'img1'), opt.img_size)
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))
        meta_info = open(os.path.join(data_root, seq, 'seqinfo.ini')).read()
        frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])
        nf, ta, tc = eval_seq(opt, dataloader, data_type, result_filename,
                              save_dir=output_dir, show_image=show_image, frame_rate=frame_rate)
        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)

        # eval
        logger.info('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(data_root, seq, data_type)
        accs.append(evaluator.eval_file(result_filename))
        if save_videos:
            output_video_path = osp.join(output_dir, '{}.mp4'.format(seq))
            cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(output_dir, output_video_path)
            os.system(cmd_str)
    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))

    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    Evaluator.save_summary(summary, os.path.join(result_root, 'summary_{}.xlsx'.format(exp_name)))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    opt = opts().init()

    if not opt.val_mot16:
        seqs_str = '''KITTI-13
                      KITTI-17
                      ADL-Rundle-6
                      PETS09-S2L1
                      TUD-Campus
                      TUD-Stadtmitte'''
        #seqs_str = '''TUD-Campus'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/train')
    else:
        seqs_str = '''MOT16-02
                      MOT16-04
                      MOT16-05
                      MOT16-09
                      MOT16-10
                      MOT16-11
                      MOT16-13'''
        data_root = os.path.join(opt.data_dir, 'MOT16/train')
    if opt.test_mot16:
        seqs_str = '''MOT16-01
                      MOT16-03
                      MOT16-06
                      MOT16-07
                      MOT16-08
                      MOT16-12
                      MOT16-14'''
        #seqs_str = '''MOT16-01 MOT16-07 MOT16-12 MOT16-14'''
        #seqs_str = '''MOT16-06 MOT16-08'''
        data_root = os.path.join(opt.data_dir, 'MOT16/test')
    if opt.test_mot15:
        seqs_str = '''ADL-Rundle-1
                      ADL-Rundle-3
                      AVG-TownCentre
                      ETH-Crossing
                      ETH-Jelmoli
                      ETH-Linthescher
                      KITTI-16
                      KITTI-19
                      PETS09-S2L2
                      TUD-Crossing
                      Venice-1'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/test')
    if opt.test_mot17:
        seqs_str = '''MOT17-01-SDP
                      MOT17-03-SDP
                      MOT17-06-SDP
                      MOT17-07-SDP
                      MOT17-08-SDP
                      MOT17-12-SDP
                      MOT17-14-SDP'''
        data_root = os.path.join(opt.data_dir, 'MOT17/images/test')
    if opt.val_mot17:
        seqs_str = '''MOT17-02-SDP
                      MOT17-04-SDP
                      MOT17-05-SDP
                      MOT17-09-SDP
                      MOT17-10-SDP
                      MOT17-11-SDP
                      MOT17-13-SDP'''
        data_root = os.path.join(opt.data_dir, 'MOT17/images/train')
    if opt.val_mot15:
        seqs_str = '''Venice-2
                      KITTI-13
                      KITTI-17
                      ETH-Bahnhof
                      ETH-Sunnyday
                      PETS09-S2L1
                      TUD-Campus
                      TUD-Stadtmitte
                      ADL-Rundle-6
                      ADL-Rundle-8
                      ETH-Pedcross2
                      TUD-Stadtmitte'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/train')
    if opt.val_mot20:
        seqs_str = '''MOT20-01
                      MOT20-02
                      MOT20-03
                      MOT20-05
                      '''
        data_root = os.path.join(opt.data_dir, 'MOT20/images/train')
    if opt.test_mot20:
        seqs_str = '''MOT20-04
                      MOT20-06
                      MOT20-07
                      MOT20-08
                      '''
        data_root = os.path.join(opt.data_dir, 'MOT20/images/test')
    seqs = [seq.strip() for seq in seqs_str.split()]

    main(opt,
         data_root=data_root,
         seqs=seqs,
         exp_name='MOT17_test_public_dla34',
         show_image=False,
         save_images=False,
         save_videos=True)
