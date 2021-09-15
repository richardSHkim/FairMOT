from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import logging
import os
import os.path as osp
import glob
from opts import opts
from tracking_utils.utils import mkdir_if_missing
from tracking_utils.log import logger
import datasets.dataset.jde as datasets
from track import eval_seq_mtmct
import cv2

# from fastreid.utils.logger import setup_logger
# setup_logger(name="fastreid")
# logger = logging.getLogger('fastreid.visualize_result')
logger.setLevel(logging.INFO)


def demo(opt):
    result_root = opt.output_root if opt.output_root != '' else '.'
    mkdir_if_missing(result_root)

    logger.info('Starting tracking...')
    dataloaders = {}
    # cnt = 0
    for i, input_vid in enumerate(sorted(list(glob.glob(os.path.join(opt.input_videos_dir, '*.mp4'))))):
        print(input_vid)
        dataloaders[i] = datasets.LoadVideo(input_vid, opt.img_size)
        # cnt += 1
        # if cnt==2:
        #     break
    result_filename = os.path.join(result_root, 'results.txt')
    frame_rate = dataloaders[list(dataloaders.keys())[0]].frame_rate
    vw = dataloaders[list(dataloaders.keys())[0]].vw
    vh = dataloaders[list(dataloaders.keys())[0]].vh

    query = cv2.imread(opt.query)
    qw = vw - int(vw/11)*10
    qh = int(query.shape[0] * (qw / query.shape[1]))
    sh = int(qh/10)

    if opt.output_format == 'video':
        output_video_path = osp.join(result_root, opt.output_name)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, 41.0, (vw, vh + qh + sh))

    frame_dir = None if opt.output_format == 'text' else osp.join(result_root, 'frame')
    eval_seq_mtmct(opt, dataloaders, 'mot', result_filename,
                      save_dir=frame_dir, show_image=False, frame_rate=frame_rate,
                      use_cuda=opt.gpus!=[-1], out=out, vw=vw, vh=vh)

    if opt.output_format == 'video':
        out.release()
        # output_video_path = osp.join(result_root, 'MOT16-03-results.mp4')
        # cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -b 5000k -c:v mpeg4 {}'.format(osp.join(result_root, 'frame'), output_video_path)
        # os.system(cmd_str)


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init()
    demo(opt)
