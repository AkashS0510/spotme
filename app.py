from fastapi import FastAPI, UploadFile, File
from typing import List, Dict
from deepface import DeepFace
import numpy as np
import uvicorn
from PIL import Image
import io

app = FastAPI()

def generate_embeddings(image):
    embeddings = []

    embedding_objs = DeepFace.represent(
    img_path = image,
    detector_backend='mtcnn',
    model_name='Facenet512',
    align=True,
    enforce_detection=False
    )
    
    for embedding_obj in embedding_objs:
        embedding = embedding_obj["embedding"]
        embeddings.append(embedding)

    numpy_embeddings = [np.array(embedding) for embedding in embeddings]

    return numpy_embeddings

def embeddings_dict(images):
    embeddings = {}
    for key, value in images.items():
        embeddings[key] = generate_embeddings(value)
        print(f"Embeddings for {key} generated")
    return embeddings

def convert_numpy_arrays_to_lists(data):
    # Iterate through the dictionary and convert numpy arrays to lists
    for key, value in data.items():
        data[key] = [arr.tolist() if isinstance(arr, np.ndarray) else arr for arr in value]
    return data

@app.post("/generate-embeddings")
async def generate_embeddings_endpoint():
    images = {
        "image1": "DSC_0212.JPG",
        "image2": "DSC_0799.JPG",
        "image3": "DSC_0823.JPG",
        "image4": "DSC_0829.JPG",
        "image5": "DSC_0883.JPG",


    }
    embeddings = embeddings_dict(images)
    embeddings = convert_numpy_arrays_to_lists(embeddings)
    return embeddings


