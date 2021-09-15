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
from track import eval_seq_query
import cv2

# from fastreid.utils.logger import setup_logger
# setup_logger(name="fastreid")
# logger = logging.getLogger('fastreid.visualize_result')
logger.setLevel(logging.INFO)


def demo(opt):
    result_root = opt.output_root if opt.output_root != '' else '.'
    mkdir_if_missing(result_root)

    logger.info('Starting tracking...')
    dataloader = datasets.LoadVideo(opt.input_video, opt.img_size)
    result_filename = os.path.join(result_root, 'results.txt')
    frame_rate = dataloader.frame_rate

    if opt.output_format == 'video':
        output_video_path = osp.join(result_root, opt.output_name)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, 41.0, (1920, 1080))

    frame_dir = None if opt.output_format == 'text' else osp.join(result_root, 'frame')
    eval_seq_query(opt, dataloader, 'mot', result_filename,
                      save_dir=frame_dir, show_image=False, frame_rate=frame_rate,
                      use_cuda=opt.gpus!=[-1], out=out)

    if opt.output_format == 'video':
        out.release()
        # output_video_path = osp.join(result_root, 'MOT16-03-results.mp4')
        # cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -b 5000k -c:v mpeg4 {}'.format(osp.join(result_root, 'frame'), output_video_path)
        # os.system(cmd_str)


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init()
    demo(opt)
