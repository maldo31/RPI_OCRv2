import base64
import io
import uuid
from PIL import Image


from flask import Flask, render_template, request,jsonify
import os
from deeplearn import OCR, text_reck


app = Flask(__name__)


BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH,'static/upload/')
@app.route('/',methods=['POST','GET'])
def index():
    if request.method == 'POST':
        upload_file = request.files['image_name']
        filename = upload_file.filename
        path_save = os.path.join(UPLOAD_PATH,filename)
        upload_file.save(path_save)
        text = OCR(path_save,filename)
        return render_template('index.html',upload=True,upload_image=filename,text=text)

    return render_template('index.html',upload=False)

@app.route('/home')
def home():
    return render_template('layout.html')

@app.route('/photo', methods=['POST'])
def register_new():
    print("New post request photo")
    img = Image.open(request.files['file'])
    filename = "photo"+str(uuid.uuid4()) + ".jpg"
    path_save = os.path.join(UPLOAD_PATH,filename)
    print(path_save)
    img.save(path_save)
    text = OCR(path_save,filename)
    print(text)
    return text

@app.route('/post', methods=['POST'])
def post():
    print(request.data)
    return ''


@app.route('/readtext', methods=['POST'])
def register_new_text():
    print("New post request readtext")
    image = request.files["image"]
    image_bytes = Image.open(io.BytesIO(image.read()))
    filename = "text"+str(uuid.uuid4()) + ".jpg"
    path_save = os.path.join(UPLOAD_PATH,filename)
    print(path_save)
    image_bytes.save(path_save)
    text = text_reck(path_save,filename)
    return text


if __name__=="__main__":
    app.run(host='0.0.0.0')