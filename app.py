from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model("pokemon_model.h5")
class_names = sorted(os.listdir("static/uploads/classes")) if os.path.exists("static/uploads/classes") else sorted(os.listdir("../image_classifier/dataset"))

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            img = image.load_img(filepath, target_size=(128, 128))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0) / 255.0

            preds = model.predict(x)
            pred_class = class_names[np.argmax(preds)]
            confidence = np.max(preds)

            return render_template("index.html", filename=file.filename, result=pred_class, confidence=confidence)

    return render_template("index.html", filename=None)

@app.route("/display/<filename>")
def display_image(filename):
    return f'<img src="/static/uploads/{filename}" width="300">'

if __name__ == "__main__":
    app.run(debug=True)
