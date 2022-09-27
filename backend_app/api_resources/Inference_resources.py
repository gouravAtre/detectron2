# -- coding: utf-8 --
'''
Created on 27-September-2022 21:29
Project: gourav 
@author: Gourav Atre
@email: gouravkumar1815@gmail.com
'''



from flask_restful import Resource
# from src.api_resources.files import UploadFiles
# from src.utils.file_utils.Filehandler import FileLoader
# from configs.inference.app_config import UPLOAD_DIR, LOADED_MODELS, RESULTS_DF, DETECTION_THRESHOLD, OUTPUT_DIR
from flask import Flask, request, jsonify, send_file
from flask import Response
from PIL import Image
from libs.mask_rcnn_detection import MaskRCNNDetection
import numpy as np
from numpy import asarray
import jsonpickle
import cv2
import flask 

class GetSegmentation(Resource):

    @classmethod
    def post(cls):
        
        print('request.files[')

        r = request
        nparr = np.fromstring(r.data, np.uint8)

        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # file = request.files['image']
        # print('Image.open')
        # img = Image.open(file.stream)

        # print('asarray(img)')
        # np_img = asarray(img)

        mcd_results = MaskRCNNDetection.segment_object(img)
        # response = flask.make_response(mcd_results['segmentation_masks'].tobytes())

        # data = cv2.imencode('.png', img)[1].tobytes()
        # response.headers.set('Content-Type', 'application/octet-stream')
        # response_pickled  = jsonpickle.encode( mcd_results['segmentation_masks'] )
        
        print('SEGMENTATTION MASKS::::\n',mcd_results['segmentation_masks'][0])
        # return Response(response=response_pickled, status=200, mimetype="application/json") 
        # return response

        _, frame  = cv2.imencode('.jpg', mcd_results['segmentation_masks'][0])
        response_pickled  = jsonpickle.encode( frame )
        return Response(response=response_pickled, status=200, mimetype="application/json") 

        # return response(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + data + b'\r\n\r\n', mimetype='multipart/x-mixed-replace; boundary=frame')