from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from tensorflow import keras

app=FastAPI()


MODEL = tf.keras.models.load_model("my_model.keras")
CLASS_NAMES=["Early Blight","Late Blight","Healthy"]

def read_file_as_image(data)->np.ndarray:
    image = np.array(Image.open(io.BytesIO(data)))
    return image

@app.get('/')
async def ping():
    return "hello world"

@app.post('/predict')
async def predict(
    file:UploadFile=File(...),
):
    bytes=await file.read()
    image= read_file_as_image(bytes)
    # print(image)
    img_batch=np.expand_dims(image,axis=0)
    
    prediction=MODEL.predict(img_batch)
    
    print(np.argmax(prediction[0]))
    
    return {
        "class":
        CLASS_NAMES[np.argmax(prediction[0])],
        "confidence":float(np.max(prediction[0]))
        }
    
    # return
    
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)