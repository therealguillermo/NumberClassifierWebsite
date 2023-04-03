from flask import Flask, request, jsonify
from base64 import b64decode
import time
from flask_cors import CORS
from nn.Net import getImagePred

app = Flask(__name__)
cors = CORS(app)
cors = CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})


@app.route('/upload-image', methods=['POST'])
def upload_image():
    data = request.get_json()
    image_data = b64decode(data['image'])
    print("dub")
    with open('image.png', 'wb') as f:
        f.write(image_data)
    time.sleep(5)
    pred = getImagePred('image.png')
    data = {'answer': pred}
    print(f"pred: {pred}")
    response = jsonify(data)
    response.headers.add('Content-Type', 'application/json')
    return response

if __name__ == '__main__':
    app.run(port=5000)