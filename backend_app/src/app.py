# -- coding: utf-8 --
'''
Created on 27-September-2022 21:09
Project: volumetric detectron2 api inference 
@author: Gourav Atre
@email: gouravkumar1815@gmail.com
'''

# import the necessary packages and modules
from flask import Flask, jsonify
from flask_restful import Api
from flask_cors import CORS
from marshmallow import ValidationError
from configs.app_config import FlaskAppConfiguration

from flask import send_from_directory
from api_resources.Inference_resources import GetSegmentation
from libs.mask_rcnn_detection import MaskRCNNDetection

# Initializing flask app...
app = Flask(__name__)


# Flask app configure and run
app.config.from_object("configs.app_config.FlaskAppConfiguration")
api = Api(app)
# cors = CORS(app, resources={r"/*": {"origins": "*"}})
cors = CORS(app, resources={r"/*": {"origins": "*", "allow_headers": "*", "expose_headers": "*"}})


# Handling all errors and returning to the frontend
@app.errorhandler(ValidationError)
def handle_marshmallow_validation(err):
    return jsonify(err.messages), 400


MaskRCNNDetection.load_model()

# MAPPING END-POINTS WITH FUNCTION LOGIC
# /admin
# api.add_resource(UploadFiles, "/upload_files/<string:video_file>/<string:excel_file>")

# api.add_resource(CheckModelsUpload, "/model_test")

api.add_resource(GetSegmentation, "/get_segmentation")


@app.route('/data/<path:path>')
def send_report(path):
    print(path)
    return send_from_directory('../data', path)