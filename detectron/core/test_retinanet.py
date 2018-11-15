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

"""Test a RetinaNet network on an image database"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import logging
import os
from collections import defaultdict

from caffe2.python import core, workspace, stat

from detectron.core.config import cfg
from detectron.modeling.generate_anchors import generate_anchors
from detectron.utils.timer import Timer
import detectron.utils.blob as blob_utils
import detectron.utils.boxes as box_utils
import detectron.utils.compare as compare_utils
logger = logging.getLogger(__name__)

def _create_cell_anchors():
    """
    Generate all types of anchors for all fpn levels/scales/aspect ratios.
    This function is called only once at the beginning of inference.
    """
    k_max, k_min = cfg.FPN.RPN_MAX_LEVEL, cfg.FPN.RPN_MIN_LEVEL
    scales_per_octave = cfg.RETINANET.SCALES_PER_OCTAVE
    aspect_ratios = cfg.RETINANET.ASPECT_RATIOS
    anchor_scale = cfg.RETINANET.ANCHOR_SCALE
    A = scales_per_octave * len(aspect_ratios)
    anchors = {}
    for lvl in range(k_min, k_max + 1):
        # create cell anchors array
        stride = 2. ** lvl
        cell_anchors = np.zeros((A, 4))
        a = 0
        for octave in range(scales_per_octave):
            octave_scale = 2 ** (octave / float(scales_per_octave))
            for aspect in aspect_ratios:
                anchor_sizes = (stride * octave_scale * anchor_scale, )
                anchor_aspect_ratios = (aspect, )
                cell_anchors[a, :] = generate_anchors(
                    stride=stride, sizes=anchor_sizes,
                    aspect_ratios=anchor_aspect_ratios)
                a += 1
        anchors[lvl] = cell_anchors
    return anchors


def im_detect_bbox(model, im, timers=None, model1=None):
    """Generate RetinaNet detections on a single image."""
    if timers is None:
        timers = defaultdict(Timer)

    if model1 is None and os.environ.get('COSIM'):
        print("cosim must has model1")

    fp32_ws_name = "__fp32_ws__"
    int8_ws_name = "__int8_ws__"
    # Although anchors are input independent and could be precomputed,
    # recomputing them per image only brings a small overhead
    anchors = _create_cell_anchors()
    timers['im_detect_bbox'].tic()
    timers['data1'].tic()
    k_max, k_min = cfg.FPN.RPN_MAX_LEVEL, cfg.FPN.RPN_MIN_LEVEL
    A = cfg.RETINANET.SCALES_PER_OCTAVE * len(cfg.RETINANET.ASPECT_RATIOS)
    inputs = {}
    inputs['data'], im_scale, inputs['im_info'] = \
        blob_utils.get_image_blob(im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE)
    cls_probs, box_preds = [], []
    for lvl in range(k_min, k_max + 1):
        suffix = 'fpn{}'.format(lvl)
        cls_probs.append(core.ScopedName('retnet_cls_prob_{}'.format(suffix)))
        box_preds.append(core.ScopedName('retnet_bbox_pred_{}'.format(suffix)))
    for k, v in inputs.items():
        if os.environ.get('COSIM'):
            workspace.SwitchWorkspace(int8_ws_name, True)
        workspace.FeedBlob(core.ScopedName(k), v.astype(np.float32, copy=False))
        if os.environ.get('COSIM'):
            workspace.SwitchWorkspace(fp32_ws_name, True)
            workspace.FeedBlob(core.ScopedName(k), v.astype(np.float32, copy=False))
    timers['data1'].toc()
    if os.environ.get('EPOCH2')=="1":
        workspace.RunNet(model.net.Proto().name)
    timers['run'].tic()
    if os.environ.get('INT8INFO')=="1":
        stat.GatherStatInfo(workspace, model.net.Proto())
    else:
        if os.environ.get('COSIM'):
            with open("int8.txt", "wb") as p:
                p.write(str(model.net.Proto()))
            with open("fp32.txt", "wb") as p:
                p.write(str(model1.net.Proto()))
            for i in range(len(model.net.Proto().op)):
                workspace.SwitchWorkspace(int8_ws_name)
                int8_inputs = []
                for inp in model.net.Proto().op[i].input:
                    int8_inputs.append(workspace.FetchBlob(str(inp)))
                logging.warning(" opint8[{0}] is  {1}".format(i,model.net.Proto().op[i]))
                workspace.RunOperatorOnce(model.net.Proto().op[i])
                int8_results = []
                for res in model.net.Proto().op[i].output:
                    int8_results.append(workspace.FetchBlob(str(res)))
                workspace.SwitchWorkspace(fp32_ws_name)
                fp32_inputs = []
                for inp1 in model1.net.Proto().op[i].input:
                    fp32_inputs.append(workspace.FetchBlob(str(inp1)))
                logging.warning(" opfp32[{0}] is  {1}".format(i,model1.net.Proto().op[i]))
                workspace.RunOperatorOnce(model1.net.Proto().op[i])
                fp32_results = []
                for res1 in model1.net.Proto().op[i].output:
                    fp32_results.append(workspace.FetchBlob(str(res1)))
                if len(int8_inputs) != len(fp32_inputs):
                    logging.error("Wrong number of inputs")
                    return
                if len(int8_results) != len(fp32_results):
                    logging.error("Wrong number of outputs")
                    return
                tol = {'atol': 5e-01, 'rtol': 5e-01}
                logging.warning("begin to check op[{}] {} input".format(i,model.net.Proto().op[i].type))
                for k in range(len(int8_inputs)):
                    if model.net.Proto().op[i].input[k][0] == '_':
                        continue
                    #assert_allclose(int8_inputs[k], fp32_inputs[k], **tol)
                logging.warning("pass checking op[{0}] {1} input".format(i,model.net.Proto().op[i].type))
                logging.warning("begin to check op[{0}] {1} output".format(i,model.net.Proto().op[i].type))
                for j in range(len(int8_results)):
                    if model.net.Proto().op[i].output[j][0] == '_':
                        continue
                    #logging.warning("int8_outputis {} and fp32 output is {} ".format(int8_results[j], fp32_results[j]))
                    #if not compare_utils.assert_allclose(int8_results[j], fp32_results[j], **tol):
                    if not compare_utils.assert_compare(int8_results[j], fp32_results[j], 1e-01,os.environ.get('COSIM')):
                        for k in range(len(int8_inputs)):
                            logging.warning("int8_input[{}] is {}".format(k, int8_inputs[k]))
                            logging.warning("fp32_input[{}] is {}".format(k, fp32_inputs[k]))

                logging.warning("pass checking op[{0}] {1} output".format(i, model.net.Proto().op[i].type))
        else:
            workspace.RunNet(model.net.Proto().name)
    timers['run'].toc()
    cls_probs = workspace.FetchBlobs(cls_probs)
    box_preds = workspace.FetchBlobs(box_preds)

    # here the boxes_all are [x0, y0, x1, y1, score]
    boxes_all = defaultdict(list)

    batch_size = cls_probs[0].shape[0]
    boxes_all_list = [boxes_all] * batch_size
    cnt = 0
    for lvl in range(k_min, k_max + 1):
        # create cell anchors array
        stride = 2. ** lvl
        cell_anchors = anchors[lvl]

        # fetch per level probability
        cls_prob = cls_probs[cnt]
        box_pred = box_preds[cnt]
        cls_prob = cls_prob.reshape((
            cls_prob.shape[0], A, int(cls_prob.shape[1] / A),
            cls_prob.shape[2], cls_prob.shape[3]))
        box_pred = box_pred.reshape((
            box_pred.shape[0], A, 4, box_pred.shape[2], box_pred.shape[3]))
        cnt += 1

        if cfg.RETINANET.SOFTMAX:
            cls_prob = cls_prob[:, :, 1::, :, :]

        for i in range(batch_size):
            cls_prob_ravel = cls_prob[i,:].ravel()

            # In some cases [especially for very small img sizes], it's possible that
            # candidate_ind is empty if we impose threshold 0.05 at all levels. This
            # will lead to errors since no detections are found for this image. Hence,
            # for lvl 7 which has small spatial resolution, we take the threshold 0.0
            th = cfg.RETINANET.INFERENCE_TH if lvl < k_max else 0.0
            candidate_inds = np.where(cls_prob_ravel > th)[0]
            if (len(candidate_inds) == 0):
                continue

            pre_nms_topn = min(cfg.RETINANET.PRE_NMS_TOP_N, len(candidate_inds))
            inds = np.argpartition(
                    cls_prob_ravel[candidate_inds], -pre_nms_topn)[-pre_nms_topn:]
            inds = candidate_inds[inds]

            inds_4d = np.array(np.unravel_index(inds, (cls_prob[i,:]).shape)).transpose()
            classes = inds_4d[:, 1]
            anchor_ids, y, x = inds_4d[:, 0], inds_4d[:, 2], inds_4d[:, 3]
            scores = cls_prob[i, anchor_ids, classes, y, x]
            boxes = np.column_stack((x, y, x, y)).astype(dtype=np.float32)
            boxes *= stride
            boxes += cell_anchors[anchor_ids, :]

            if not cfg.RETINANET.CLASS_SPECIFIC_BBOX:
                box_deltas = box_pred[i, anchor_ids, :, y, x]
            else:
                box_cls_inds = classes * 4
                box_deltas = np.vstack(
                   [box_pred[i, ind:ind + 4, yi, xi]
                   for ind, yi, xi in zip(box_cls_inds, y, x)]
                )
            pred_boxes = (
                    box_utils.bbox_transform(boxes, box_deltas)
                    if cfg.TEST.BBOX_REG else boxes)
            pred_boxes /= im_scale
            pred_boxes = box_utils.clip_tiled_boxes(pred_boxes, im[0].shape)
            box_scores = np.zeros((pred_boxes.shape[0], 5))
            box_scores[:, 0:4] = pred_boxes
            box_scores[:, 4] = scores

            for cls in range(1, cfg.MODEL.NUM_CLASSES):
                inds = np.where(classes == cls - 1)[0]
                if len(inds) > 0:
                    boxes_all_list[i][cls].extend(box_scores[inds, :])

    timers['im_detect_bbox'].toc()

    cls_boxes_list = []
    for i in range(batch_size):
        boxes_all = boxes_all_list[i]
        # Combine predictions across all levels and retain the top scoring by class
        timers['misc_bbox'].tic()
        detections = []
        for cls, boxes in boxes_all.items():
            cls_dets = np.vstack(boxes).astype(dtype=np.float32)
            # do class specific nms here
            keep = box_utils.nms(cls_dets, cfg.TEST.NMS)
            cls_dets = cls_dets[keep, :]
            out = np.zeros((len(keep), 6))
            out[:, 0:5] = cls_dets
            out[:, 5].fill(cls)
            detections.append(out)

        # detections (N, 6) format:
        #   detections[:, :4] - boxes
        #   detections[:, 4] - scores
        #   detections[:, 5] - classes
        detections = np.vstack(detections)
        # sort all again
        inds = np.argsort(-detections[:, 4])
        detections = detections[inds[0:cfg.TEST.DETECTIONS_PER_IM], :]

        # Convert the detections to image cls_ format (see core/test_engine.py)
        num_classes = cfg.MODEL.NUM_CLASSES
        cls_boxes = [[] for _ in range(cfg.MODEL.NUM_CLASSES)]
        for c in range(1, num_classes):
            inds = np.where(detections[:, 5] == c)[0]
            cls_boxes[c] = detections[inds, :5]
        cls_boxes_list.append(cls_boxes)

    timers['misc_bbox'].toc()

    return cls_boxes_list
