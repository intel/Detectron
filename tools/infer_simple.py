#!/usr/bin/env python2

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import glob
import logging
import os
import sys
import time
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
from caffe2.python import workspace

from detectron.core.calibrator import Calibrator, KLCalib, AbsmaxCalib, EMACalib
from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils

c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    """
    to parse the argument
    """
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='/tmp/infer_simple',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        '--always-out',
        dest='out_when_no_box',
        help='output image even when no object is found',
        action='store_true'
    )
    parser.add_argument(
        'im_or_folder', help='image or folder of images', default=None
    )
    parser.add_argument(
        '--output-ext',
        dest='output_ext',
        help='output image file format (default: pdf)',
        default='pdf',
        type=str
    )
    parser.add_argument(
        '--device_id',
        dest='device_id',
        default=0,
        type=int
    )
    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        default=1,
        type=int
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def batch_image(im_list, batch_size):
    """
    to put the image into batches
    """
    bs = batch_size
    fnames = []
    fname = []
    for _, im_name in enumerate(im_list):
        bs -= 1
        fname.append(im_name)
        if bs == 0:
            fnames.append(fname)
            fname = []
            bs = batch_size
    if len(fname) > 0:
        fnames.append(fname)

    return fnames

def main(args):
    """
    main entry to run
    """
    logger = logging.getLogger(__name__)

    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)

    assert not cfg.MODEL.RPN_ONLY, \
        'RPN models are not supported'
    assert not cfg.TEST.PRECOMPUTED_PROPOSALS, \
        'Models that require precomputed proposals are not supported'
    fp32_ws_name = "__fp32_ws__"
    int8_ws_name = "__int8_ws__"
    model1 = None
    if os.environ.get('COSIM'):
        workspace.SwitchWorkspace(int8_ws_name, True)
    model, _, _, _ = infer_engine.initialize_model_from_cfg(args.weights, gpu_id=args.device_id)
    if os.environ.get('COSIM'):
        workspace.SwitchWorkspace(fp32_ws_name, True)
        model1, _, _, _ = infer_engine.initialize_model_from_cfg(args.weights, gpu_id=args.device_id, int8=False)

    dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    if os.path.isdir(args.im_or_folder):
        im_list = glob.iglob(args.im_or_folder + '/*.' + args.image_ext)
    else:
        im_list = [args.im_or_folder]

    fnames = batch_image(im_list, args.batch_size)
    # for kl_divergence calibration, we use the first 100 images to get
    # the min and max values, and the remaing images are applied to compute the hist.
    # if the len(images) <= 100, we extend the images with themselves.
    if os.environ.get('INT8INFO') == "1" and os.environ.get('INT8CALIB') == "kl_divergence":
        kl_iter_num_for_range = os.environ.get('INT8KLNUM')
        if not kl_iter_num_for_range:
            kl_iter_num_for_range = 100
        kl_iter_num_for_range = int(kl_iter_num_for_range)
        while (len(fnames) < 2*kl_iter_num_for_range):
            fnames += fnames
    if os.environ.get('EPOCH2') == "1":
        for i, im_name in enumerate(fnames):
            im = []
            for _, name in enumerate(im_name):
                image = cv2.imread(name)
                im.append(image)

            timers = defaultdict(Timer)
            t = time.time()
            with c2_utils.NamedCudaScope(args.device_id):
                cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                    model, im, None, timers, model1
                )
    logger.warning("begin to run benchmark\n")
    for i, im_name in enumerate(fnames):
        im = []
        for _, name in enumerate(im_name):
            image = cv2.imread(name)
            im.append(image)

        timers = defaultdict(Timer)
        t = time.time()
        with c2_utils.NamedCudaScope(args.device_id):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                model, im, None, timers, model1
            )
        logger.info('Inference time: {:.3f}s'.format(time.time() - t))
        for k, v in timers.items():
            logger.info(' | {}: {:.3f}s'.format(k, v.average_time))

        if i == 0:
            logger.info(
                ' \ Note: inference on the first batch will be slower than the '
                'rest (caches and auto-tuning need to warm up)'
            )

        cls_segm = None
        cls_keyp = None
        for bs in range(args.batch_size):
            image = im[bs]
            if cls_segms != None:
                cls_segm = cls_segms[bs]
            if cls_keyp != None:
                cls_keyp = cls_keyps[bs]

            cls_box = cls_boxes[bs]
            image_name = fnames[i][bs].split("/")[-1]

            vis_utils.vis_one_image(
                image[:, :, ::-1],  # BGR -> RGB for visualization
                image_name,
                args.output_dir,
                cls_box,
                cls_segm,
                cls_keyp,
                dataset=dummy_coco_dataset,
                box_alpha=0.3,
                show_class=True,
                thresh=0.7,
                kp_thresh=2,
                ext=args.output_ext,
                out_when_no_box=args.out_when_no_box
            )

    if os.environ.get('INT8INFO') == "1":
        def save_net(net_def, init_def):
            if net_def is None or init_def is None:
                return
            if net_def.name is None or init_def.name is None:
                return
            if os.environ.get('INT8PTXT') == "1":
                with open(net_def.name + '_predict_int8.ptxt', 'wb') as n:
                    n.write(str(net_def))
                with open(net_def.name + '_init_int8.ptxt', 'wb') as n:
                    n.write(str(init_def))
            else:
                with open(net_def.name + '_predict_int8.pb', 'wb') as n:
                    n.write(net_def.SerializeToString())
                with open(net_def.name + '_init_int8.pb', 'wb') as n:
                    n.write(init_def.SerializeToString())
        algorithm = AbsmaxCalib()
        kind = os.environ.get('INT8CALIB')
        if kind == "moving_average":
            ema_alpha = 0.5
            algorithm = EMACalib(ema_alpha)
        elif kind == "kl_divergence":
            algorithm = KLCalib(kl_iter_num_for_range)
        calib = Calibrator(algorithm)
        if model.net:
            predict_quantized, init_quantized = calib.DepositQuantizedModule(workspace, model.net.Proto())
            save_net(predict_quantized, init_quantized)
        if cfg.MODEL.MASK_ON:
            predict_quantized, init_quantized = calib.DepositQuantizedModule(workspace, model.mask_net.Proto())
            save_net(predict_quantized, init_quantized)
        if cfg.MODEL.KEYPOINTS_ON:
            predict_quantized, init_quantized = calib.DepositQuantizedModule(workspace, model.keypoint_net.Proto())
            save_net(predict_quantized, init_quantized)

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    setup_logging(__name__)
    args = parse_args()
    main(args)
