from flask import Flask, request, jsonify, render_template
from main import detect_emotion

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # If using an HTML frontend

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    img = request.files['image']
    emotion = detect_emotion(img)
    return jsonify({'emotion': emotion})

if __name__ == '__main__':
    app.run(debug=True)
