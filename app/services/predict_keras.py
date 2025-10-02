import os, json, numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = "app/models/landcover_mnv2.h5"
LABELS_PATH = "app/models/labels.json"

_model, _labels = None, None

def _load():
    global _model, _labels
    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH)
        _labels = json.load(open(LABELS_PATH))
    return _model, _labels

def predict(image: Image.Image):
    model, labels = _load()
    img = image.resize((224,224)).convert("RGB")
    arr = np.expand_dims(preprocess_input(np.array(img, np.float32)), 0)
    probs = model.predict(arr)[0]
    distribution = {lbl: round(float(p)*100,2) for lbl,p in zip(labels, probs)}
    confidences = {lbl: round(float(p),3) for lbl,p in zip(labels, probs)}
    # Overlay color by top class (placeholder)
    top = labels[int(np.argmax(probs))]
    colors = {"forest":(34,139,34,120),"water":(65,105,225,120),
              "urban":(169,169,169,120),"barren":(184,134,11,120)}
    overlay = Image.new("RGBA", image.size, colors.get(top,(255,255,255,80)))
    return {"overlay": overlay, "distribution": distribution, "confidences": confidences}
