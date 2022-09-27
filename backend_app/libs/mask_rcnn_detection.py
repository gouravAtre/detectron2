# -*- coding: utf-8 -*-
'''
Created on 27-September-2022 00:28
Project: Ai-Space 
@author: Pranjal Bhaskare, Gourav Atre
@email: pranjalab@neophyte.live, gouravkumar1815@gmail.com
'''

# Some basic setup:
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import os
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import numpy as np


class MaskRCNNDetection:

	@classmethod
	def load_model(cls):
		
		cfg = get_cfg()
		cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
		cfg.DATASETS.TRAIN = ("my_dataset_train3",)
		cfg.DATASETS.TEST = ()
		cfg.DATALOADER.NUM_WORKERS = 2
		cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
		cfg.SOLVER.IMS_PER_BATCH = 8  # This is the real "batch size" commonly known to deep learning people
		cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
		cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
		cfg.SOLVER.STEPS = []        # do not decay learning rate
		cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
		cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
		# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
		cfg.OUTPUT_DIR = './files/rcnn_models/train1_volumetric'


		# Inference should use the config with parameters that are used in training
		# cfg now already contains everything we've set previously. We changed it a little bit for inference:
		cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
		cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
		cls.predictor = DefaultPredictor(cfg)

	@classmethod
	def segment_object(cls, img):
		outputs = cls.predictor(img)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    
		segmentation_masks = outputs["instances"].pred_masks.cpu().numpy()
		segmentation_masks = np.array(segmentation_masks, dtype= np.uint8)
		return {
			"image": img,
			"segmentation_masks": segmentation_masks
		}