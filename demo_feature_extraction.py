#!/usr/bin/env python
# coding: utf-8

import os
import io

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs, fast_rcnn_inference_single_image

import numpy as np
import cv2
import torch
import streamlit as st

@st.cache
class FrcnFeatures():
    
    def __init__(self):
        
        self.cfg = get_cfg()
        #self.cfg.merge_from_file("/Users/apoorve/Desktop/Personal/streamlit_share_demo.nosync/configs/COCO-Detection/faster_rcnn_R_101_C4_3x.yaml")
        self.cfg.merge_from_file("configs/VG-Detection/faster_rcnn_R_101_C4_caffe.yaml")

        # self.cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 500
        # self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
        # self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

        self.cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
        self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
 
        self.cfg.MODEL.DEVICE = 'cpu'
        #self.cfg.MODEL.WEIGHTS = "/Users/apoorve/Downloads/model_final_298dad.pkl"
        self.cfg.MODEL.WEIGHTS = "http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe.pkl"
        self.predictor = DefaultPredictor(self.cfg)
        
        self.NUM_OBJECTS = 36
    
    def __call__(self, raw_image):

        with torch.no_grad():
            raw_height, raw_width = raw_image.shape[:2]
            print("Original image size: ", (raw_height, raw_width))

            # Preprocessing
            image = self.predictor.transform_gen.get_transform(raw_image).apply_image(raw_image)
            print("Transformed image size: ", image.shape[:2])
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = [{"image": image, "height": raw_height, "width": raw_width}]
            images = self.predictor.model.preprocess_image(inputs)

            # Run Backbone Res1-Res4
            features = self.predictor.model.backbone(images.tensor)

            # Generate proposals with RPN
            proposals, _ = self.predictor.model.proposal_generator(images, features, None)
            proposal = proposals[0]
            print('Proposal Boxes size:', proposal.proposal_boxes.tensor.shape)

            # Run RoI head for each proposal (RoI Pooling + Res5)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            features = [features[f] for f in self.predictor.model.roi_heads.in_features]
            box_features = self.predictor.model.roi_heads._shared_roi_transform(
                features, proposal_boxes
            )
            feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
            print('Pooled features size:', feature_pooled.shape)

            # Predict classes and boxes for each proposal.
            pred_class_logits, pred_proposal_deltas = self.predictor.model.roi_heads.box_predictor(feature_pooled)
            outputs = FastRCNNOutputs(
                self.predictor.model.roi_heads.box2box_transform,
                pred_class_logits,
                pred_proposal_deltas,
                proposals,
                self.predictor.model.roi_heads.smooth_l1_beta,
            )
            probs = outputs.predict_probs()[0]
            boxes = outputs.predict_boxes()[0]

            # Note: BUTD uses raw RoI predictions,
            #       we use the predicted boxes instead.
            # boxes = proposal_boxes[0].tensor    

            # NMS
            for nms_thresh in np.arange(0.5, 1.0, 0.1):
                instances, ids = fast_rcnn_inference_single_image(
                    boxes, probs, image.shape[1:], 
                    score_thresh=0.2, nms_thresh=nms_thresh, topk_per_image=self.NUM_OBJECTS
                )
                if len(ids) == self.NUM_OBJECTS:
                    break

            instances = detector_postprocess(instances, raw_height, raw_width)

            # Scale bounding boxes to [0,1]
            instances.pred_boxes.tensor /= (1.0*raw_height)

            roi_features = feature_pooled[ids].detach()
            print(instances)

            return instances.pred_boxes.tensor, roi_features


if __name__ == '__main__':
    im = cv2.imread("assets/input.jpg")
    im = cv2.resize(im, (224, 224))

    frcnn = FrcnFeatures()
    bbox, feat = frcnn(im)

    print(bbox.shape)
    print(feat.shape)
