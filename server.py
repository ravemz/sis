import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
from datetime import datetime
from flask import Flask, request, render_template, jsonify, abort
from pathlib import Path
import uuid
import os

app = Flask(__name__)

# Read image features
fe = FeatureExtractor()
features = []
img_paths = []
# added support for nested directories
for feature_path in Path("./static/feature").glob("**/*.npy"):
    features.append(np.load(feature_path))
    # just remove the npy extension and reverse the static folder to get the file.
    img_path = Path(str(feature_path).replace("static/feature", "static/img").replace(".npy", ""))
    img_paths.append(img_path)

features = np.array(features)

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'webp'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']

        # Save query image
        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)

        # Run search
        query = fe.extract(img)
        dists = np.linalg.norm(features-query, axis=1)  # L2 distances to features
        ids = np.argsort(dists)[:15]  # Top 15 results
        scores = [(dists[id], img_paths[id]) for id in ids]

        return render_template('index.html',
                               query_path=uploaded_img_path,
                               scores=scores)
    else:
        return render_template('index.html')

@app.route('/api/search', methods=['POST'])
def search():
    # print(request)
    file = request.files['file']

    if file.filename == '':
        error = 'No file selected for uploading'
        return jsonify({"error": error}), 400

    if file and allowed_file(file.filename):
        # img = Image.open("./static/img/KEPL-compressed/KEPL-254.jpg")
        img = Image.open(file.stream)

        # generate unique filename
        filename = str(uuid.uuid4()) + '.' + file.filename.rsplit('.', 1)[1].lower()

        img.save("static/uploaded/" + filename)

        # Run search
        query = fe.extract(img)
        dists = np.linalg.norm(features-query, axis=1)  # L2 distances to features
        ids = np.argsort(dists)[:15]  # Top 15 results
        scores = [(dists[id], img_paths[id]) for id in ids]

        # Store results in a dictionary
        results = []

        for item in scores:
            results.append({
                "filepath" : str(item[1]),
                "score": str(item[0]),
                "material": 'SS304',
                "location": 'Bangalore, India',
                "process": ''
            })

        # Return a JSON response using the jsonify() function
        return jsonify(results)
    else:
        error = 'Allowed file types are png, jpg, jpeg, webp'
        return jsonify({"error": error}), 400

if __name__=="__main__":
    app.run("0.0.0.0")
