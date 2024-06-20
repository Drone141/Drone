from flask import Flask, request, jsonify, render_template, session, send_file
from PIL import Image, ExifTags, ImageDraw
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import os
from keras.models import load_model
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['CROPPED_FOLDER'] = 'C:/Users/sam00/OneDrive/Desktop/new/Online-crap'  # 新增的儲存路徑
app.config['SECRET_KEY'] = 'a very secret key'
model = load_model('C:/Users/sam00/OneDrive/Desktop/python-cnn/0615_model.h5')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['CROPPED_FOLDER'], exist_ok=True)  # 確保裁剪圖片的資料夾存在

class_names = ['lowYield', 'mediumYield', 'highYield', 'Danger', 'other']
class_weights = {
    'lowYield': 1665.026,
    'mediumYield': 1847.198,
    'highYield': 2211.542,
    'Danger': 1300.682,
    'other': 0  # No weight for 'other'
}

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        session['last_image_path'] = file_path

        image = Image.open(file_path)
        exif_data = image._getexif()
        capture_time = "No EXIF data"
        if exif_data:
            exif = {ExifTags.TAGS[k]: v for k, v in exif_data.items() if k in ExifTags.TAGS}
            capture_time = exif.get('DateTimeOriginal', 'Not Available')
        
        session['capture_time'] = capture_time
        return jsonify({"message": "File uploaded successfully", "capture_time": capture_time})

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    planting_date = request.form['planting_date']
    capture_time = session.get('capture_time', 'Not Available')
    
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        session['last_image_path'] = file_path

        results, area = split_and_predict(file_path)
        session['results'] = [int(result) for result in results]

        total_weight = sum(class_weights[class_names[result]] for result in results if class_names[result] in class_weights)
        rounded_total_weight = round(total_weight, 3)

        days_since_planting = "Unknown"
        if capture_time != "Not Available" and capture_time != "No EXIF data":
            try:
                planting_date_obj = datetime.strptime(planting_date, '%Y-%m-%d')
                capture_date_obj = datetime.strptime(capture_time.split(' ')[0], '%Y:%m:%d')
                days_since_planting = (capture_date_obj - planting_date_obj).days
            except ValueError:
                print(f"Error parsing date: {capture_time} does not match format 'YYYY:MM:DD'")

        low_yield_estimate = rounded_total_weight * 0.34 / 1000
        high_yield_estimate = rounded_total_weight * 0.43 / 1000
        rounded_total_estimate = rounded_total_weight * 0.385 / 1000
        rounded_total_weight2 = round(rounded_total_estimate, 3)
        rounded_low_yield = round(low_yield_estimate, 3)
        rounded_high_yield = round(high_yield_estimate, 3)

        return jsonify({
            "水稻預測產量 (Kg)": rounded_total_weight2,
            "capture_time": capture_time,
            "area_ha": area,
            "days_since_planting": days_since_planting,
            "low_yield_estimate": rounded_low_yield,
            "high_yield_estimate": rounded_high_yield
        })


def split_and_predict(image_path, grid_size=(28, 28), pixel_to_ha=1e-6, confidence_threshold=0.5):
    image = Image.open(image_path).convert('RGB')
    image_width, image_height = image.size
    confident_results = []
    for top in range(0, image_height, grid_size[1]):
        for left in range(0, image_width, grid_size[0]):
            right = left + grid_size[0]
            bottom = top + grid_size[1]
            cropped_image = image.crop((left, top, right, bottom))
            cropped_filename = f"cropped_{left}_{top}.png"
            cropped_path = os.path.join(app.config['CROPPED_FOLDER'], cropped_filename)
            cropped_image.save(cropped_path)  # 儲存裁剪後的圖片

            img_array = np.array(cropped_image) / 255.0
            img_array = cv2.resize(img_array, (64, 64))
            img_array = np.expand_dims(img_array, axis=0)
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction)
            max_confidence = np.max(prediction)
            if max_confidence >= confidence_threshold:
                confident_results.append(predicted_class)
    area_ha = (grid_size[0] * grid_size[1] * len(confident_results)) * pixel_to_ha
    return confident_results, area_ha

def colorize_image(image_path, results):
    image = Image.open(image_path).convert('RGB')
    width, height = image.size
    grid_size = (28, 28)
    overlay = Image.new('RGBA', image.size)
    draw = ImageDraw.Draw(overlay)
    for idx, result in enumerate(results):
        top = (idx // (width // grid_size[0])) * grid_size[1]
        left = (idx % (width // grid_size[0])) * grid_size[0]
        color_map = {
            'lowYield': (255, 255, 0, 128),   # Yellow
            'mediumYield': (255, 165, 0, 128),  # Orange
            'highYield': (0, 128, 0, 128),   # Green
            'Danger': (255, 0, 0, 128),       # Red
            'other': (128, 128, 128, 128)    # Grey
        }
        color = color_map[class_names[result]]
        draw.rectangle([left, top, left+grid_size[0], top+grid_size[1]], fill=color)
    colorized_image = Image.alpha_composite(image.convert('RGBA'), overlay)
    colorized_path = f"{image_path.split('.')[0]}_colorized.png"
    colorized_image.save(colorized_path)
    return colorized_path

@app.route('/colorized', methods=['GET'])
def colorized_result():
    image_path = session.get('last_image_path', '')
    results = session.get('results', [])
    if image_path and results:
        colorized_path = colorize_image(image_path, results)
        return send_file(colorized_path, mimetype='image/png')
    return jsonify({"error": "No results available"}), 404

@app.route('/results', methods=['GET'])
def results():
    results = session.get('results', [])
    if results:
        formatted_results = [class_names[result] for result in results]
    else:
        formatted_results = ['No results']
    return jsonify({"recognition_results": formatted_results})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
