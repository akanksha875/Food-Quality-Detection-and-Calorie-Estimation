from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from calorie_map import calorie_dict, quality_dict

app = Flask(__name__)

model = load_model('model/food11_model.h5')
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

classes = list(calorie_dict.keys())

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        f = request.files['file']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        f.save(filepath)

        # Load and preprocess the image
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict the class
        predictions = model.predict(img_array)
        predicted_class = classes[np.argmax(predictions[0])]
        calories = calorie_dict.get(predicted_class, "N/A")
        quality = quality_dict.get(predicted_class, "Nutritional information not available")

        return render_template('index.html',
                               prediction=predicted_class.replace("_", " ").title(),
                               calories=calories,
                               quality=quality,
                               user_image=filepath)

    return render_template('index.html')

# Remove app.run()

# This line tells Vercel this is your app's entry point
app = app
