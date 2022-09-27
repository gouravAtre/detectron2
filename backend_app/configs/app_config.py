# -- coding: utf-8 --
'''
Created on 09-July-2022 17:07
Project: flipkart_mis_shipment 
@author: Gourav Atre
@email: gouravkumar1815@gmail.com
'''
# import pandas as pd

# from src.app import LOADED_MODELS


class FlaskAppConfiguration:
    DEBUG = True
    PROPAGATE_EXCEPTIONS = True
    SECRET_KEY = 'change-this-key'
    
    # MODE = "Training"       # Accepted values = "Training" , "Inference"


# # Global results dictionary to append all result
# RESULTS_DF =  pd.DataFrame(columns=["WID","Scan_Time","SKU_Info", "Model_Prediction","Verification_Status", "Clip_Path" ])   

# LOADED_MODELS = dict()      # initializing this dict to save 
#                             # all loaded models globally
# DETECTION_THRESHOLD = 0.5      # for positive detections                  

# UPLOAD_DIR = "/home/gourav/projects/flipkart_mis_shipment/data/inference/demo_flipkart"
# OUTPUT_DIR = "/home/gourav/projects/flipkart_mis_shipment/data/inference/results/"

# inference_json_path = "./configs/inference/inference.json"

# # Database configuration
# DB_USERNAME = config('DB_USERNAME')
# PASSWORD = config('PASSWORD')
# DATABASE_NAME = config('DATABASE_NAME')

# DEMO_FLAG = True

# Bouding boxes while save

# infer_color = (0,255,0) 
# thickness = 7

# split_csv_path = "/home/gourav/projects/flipkart_mis_shipment/data/inference/results/Dispatched_Data_09.07.22.xlsx_split.csv"