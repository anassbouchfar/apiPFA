import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
#import pandas as pd
import cv2
#from glob import glob
#from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
#from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from metrics import dice_loss, dice_coef, iou
#from typing_extensions import Annotated
from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse

import datetime


H = 512
W = 512

def create_dir(path):
    #print(str(os.path)+path+" created")
    if not os.path.exists(path):
        os.makedirs(path)
        

def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (W, H))
    ori_x = x
    x = x/255.0
    x = x.astype(np.float32)
    return ori_x, x
"""
def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  ## (512, 512)
    x = cv2.resize(x, (W, H))
    ori_x = x
    x = x/255.0
    x = x.astype(np.int32)
    return ori_x, x"""
"""
def load_data(path):
    x = sorted(glob(os.path.join(path, "image", "*.jpg")))
    y = sorted(glob(os.path.join(path, "mask", "*.jpg")))
    print("images load : ")
    print(x)
    return x, y"""

def save_results(ori_x, y_pred, save_image_path):
    line = np.ones((H, 10, 3)) * 255

    #ori_y = np.expand_dims(ori_y, axis=-1)
    #ori_y = np.concatenate([ori_y, ori_y, ori_y], axis=-1)

    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1) * 255

    cat_images = np.concatenate([ori_x, line, y_pred], axis=1)
    cv2.imwrite(save_image_path, cat_images)

#absPath = "/Users/Shared/Relocated Items/Security/D/ENSIAS/S4/cloud project/"
#os.path = absPath
#os.chdir(absPath)
""" Save the results in this folder """
#print("create the results folder")
create_dir("results")

""" Load the model """
with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
    model = tf.keras.models.load_model("model.h5")

""" Load the dataset """
#dataset_path = os.path.join("new_data", "test")
#test_x, test_y = load_data(dataset_path)



def process(img):
    ori_x, x = read_image("input/"+img)
    # Prediction 
    y_pred = model.predict(np.expand_dims(x, axis=0))[0]
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.int32)
    y_pred = np.squeeze(y_pred, axis=-1)

    # Saving the images
    name = img.split(".")[0]
    save_image_path = f"/results/{name}.png"
    save_results(ori_x, y_pred, save_image_path)
    return save_image_path


from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.mount("/results", StaticFiles(directory="results"), name="results")

@app.get("/")
async def root():
    return {"message": "Hello World"}


# @app.post("/files/")
# async def create_file(img: bytes = File(...)):
#     return {"file_size": len(img)}


@app.post("/uploadfile/")
async def create_upload_file(img: UploadFile):
    try:
        contents = img.file.read()
        ts = str(datetime.datetime.now().timestamp())
        name  = img.filename.split(".")[0] + ts + "." + img.filename.split(".")[1]
        with open("input/"+name, 'wb') as f:
            f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        img.file.close()
    path=process(name)
    #return FileResponse(path, media_type="image/png", filename=path.split("/")[-1])
    return {"filename": path.split("/")[-1],
            "path": path
            }



"""
# Make the prediction and calculate the metrics values 
SCORE = []
img = "/Users/macbook/Downloads/RBVS-UNET-in-TensorFlow/img.jpg"
ori_x, x = read_image(img)

# Prediction 
y_pred = model.predict(np.expand_dims(x, axis=0))[0]
y_pred = y_pred > 0.5
y_pred = y_pred.astype(np.int32)
y_pred = np.squeeze(y_pred, axis=-1)

# Saving the images
save_image_path = f"results/snd/img.png"
save_results(ori_x, y_pred, save_image_path)"""

