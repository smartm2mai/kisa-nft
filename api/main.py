from PIL import Image
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import tensorflow as tf
import json
import argparse
import ipfshttpclient
import hashlib
import faiss
import json
from io import BytesIO
import datetime
import sqlite3
import base64
from fastapi import FastAPI, HTTPException, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sys.path.append("..")

from demo.image_similarity_keras.model import SiameseModel

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

model_path = "../demo/models/ConvNext_Large_64b_100ep_final"
augmentation_config = "../demo/configs/default_augmentation.json"

# Load model config
with open(os.path.join(model_path, "configs.json"), "r") as f:
    model_config = json.load(f)

    # Convert to Namespace
    model_config_ns = argparse.Namespace(**model_config)

# Load augmentation config
with open(augmentation_config, "r") as f:
    augmentation_config = json.load(f)

# Convert model_config dictionary to a namespace
model_config_ns = argparse.Namespace(**model_config)

# Get the image_size from model_config or use default value if missing
default_image_size = 224
image_size = model_config.get('image_size', default_image_size)

# Initialize model
model = SiameseModel(**model_config)

# Build and compile model
model.build(False)

# Load weights
model.model.load_weights(os.path.join(model_path, "weights"))

client = ipfshttpclient.connect(timeout=300)

conn = sqlite3.connect('nft.db')
cursor = conn.cursor()

def cid_to_int(cid):
    return int(hashlib.sha256(cid.encode()).hexdigest(), 16) % (2**31 - 1)

def reset():
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS current (
        id INTEGER PRIMARY KEY,
        hash TEXT NOT NULL
    )
    ''')
    cursor.execute("DELETE FROM current")
    cursor.execute("INSERT INTO current SELECT * FROM base")
    conn.commit()
    
    index = faiss.read_index('base.index')
    faiss.write_index(index, 'current.index')
    
    items = client.files.ls("/nft_images/new_images")['Entries']
    if items:
        for item in items:
            client.files.rm(f"/nft_images/new_images/{item['Name']}", recursive=True)

def check_and_register(img_file):
    result = {
        "plagiarism": False,
        "orig_img_cid": None,
        "orig_img_file": None,
        "probability": None,
        "new_img_cid": None
    }
    index = faiss.read_index('current.index')
    
    test_img = Image.open(BytesIO(img_file))
    test_img = test_img.convert("RGB")
    test_img = test_img.resize((224, 224))
    test_img = tf.keras.preprocessing.image.img_to_array(test_img) / 255.0
    test_img_embs = model.predict(tf.expand_dims(test_img, axis=0))
    
    distances, indices = index.search(np.array(test_img_embs), index.ntotal)
    
    if distances[0][0] < 0.52:
        result["plagiarism"] = True
        cursor.execute("SELECT hash FROM current WHERE id=?", (int(indices[0][0]),))
        orig_img_cid = cursor.fetchone()[0]
        result["orig_img_cid"] = orig_img_cid
        result["orig_img_file"] = base64.b64encode(client.cat(orig_img_cid)).decode('utf-8')
        result["probability"] = float(1 - distances[0][0]) * 100
    else:
        timestamp_str = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        new_filename = f"{timestamp_str}.png"
        client.files.write("/nft_images/new_images/" + new_filename, BytesIO(img_file), create=True)
        new_img_cid = client.files.stat("/nft_images/new_images/" + new_filename)['Hash']
        result["new_img_cid"] = new_img_cid
        index.add_with_ids(np.array(test_img_embs), np.array([cid_to_int(new_img_cid)]))
        faiss.write_index(index, 'current.index')
        cursor.execute("INSERT INTO current (id, hash) VALUES (?, ?)", (cid_to_int(new_img_cid), new_img_cid))
        conn.commit()
        
    return json.dumps(result, ensure_ascii=False)

def force_register(img_file):
    result = {
        "plagiarism": False,
        "orig_img_cid": None,
        "orig_img_file": None,
        "probability": None,
        "new_img_cid": None
    }
    index = faiss.read_index('current.index')
    
    test_img = Image.open(BytesIO(img_file))
    test_img = test_img.convert("RGB")
    test_img = test_img.resize((224, 224))
    test_img = tf.keras.preprocessing.image.img_to_array(test_img) / 255.0
    test_img_embs = model.predict(tf.expand_dims(test_img, axis=0))
    
    timestamp_str = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    new_filename = f"{timestamp_str}.png"
    client.files.write("/nft_images/new_images/" + new_filename, BytesIO(img_file), create=True)
    new_img_cid = client.files.stat("/nft_images/new_images/" + new_filename)['Hash']
    result["new_img_cid"] = new_img_cid
    index.add_with_ids(np.array(test_img_embs), np.array([cid_to_int(new_img_cid)]))
    faiss.write_index(index, 'current.index')
    cursor.execute("INSERT INTO current (id, hash) VALUES (?, ?)", (cid_to_int(new_img_cid), new_img_cid))
    conn.commit()
        
    return json.dumps(result, ensure_ascii=False)

def only_check(img_file):
    result = {
        "plagiarism": False,
        "orig_img_cid": None,
        "orig_img_file": None,
        "probability": None,
        "new_img_cid": None
    }
    index = faiss.read_index('current.index')
    
    test_img = Image.open(BytesIO(img_file))
    test_img = test_img.convert("RGB")
    test_img = test_img.resize((224, 224))
    test_img = tf.keras.preprocessing.image.img_to_array(test_img) / 255.0
    test_img_embs = model.predict(tf.expand_dims(test_img, axis=0))
    
    distances, indices = index.search(np.array(test_img_embs), index.ntotal)
    
    if distances[0][0] < 0.52:
        result["plagiarism"] = True
        cursor.execute("SELECT hash FROM current WHERE id=?", (int(indices[0][0]),))
        orig_img_cid = cursor.fetchone()[0]
        result["orig_img_cid"] = orig_img_cid
        result["orig_img_file"] = base64.b64encode(client.cat(orig_img_cid)).decode('utf-8')
        result["probability"] = float(1 - distances[0][0]) * 100
        
    return json.dumps(result, ensure_ascii=False)

class ImageInput(BaseModel):
    img_file: str

class RegisterResponse(BaseModel):
    plagiarism: bool
    orig_img_cid: Optional[str] = None
    orig_img_file: Optional[str] = None
    probability: Optional[float] = None
    new_img_cid: Optional[str] = None

@app.on_event("startup")
def startup_event():
    pass

@app.post("/reset/", response_model=None)
async def reset_endpoint():
    try:
        reset()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "reset successful"}

@app.post("/check_and_register/", response_model=RegisterResponse)
async def check_and_register_endpoint(image: ImageInput):
    try:
        img_file_bytes = base64.b64decode(image.img_file)
        result = check_and_register(img_file_bytes)
        return json.loads(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/force_register/", response_model=RegisterResponse)
async def force_register_endpoint(image: ImageInput):
    try:
        img_file_bytes = base64.b64decode(image.img_file)
        result = force_register(img_file_bytes)
        return json.loads(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/only_check/", response_model=RegisterResponse)
async def only_check_endpoint(image: ImageInput):
    try:
        img_file_bytes = base64.b64decode(image.img_file)
        result = only_check(img_file_bytes)
        return json.loads(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
