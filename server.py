import flask
import json

from flask import Flask, request
from werkzeug.utils import secure_filename
from facedb import FaceDatabase

db = FaceDatabase()
app = Flask(__name__)

@app.route("/")
def index():
    return "<p>Hello, World!</p>"

@app.route('/detect', methods = ['GET', 'POST'])
def detect():
    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(f.filename))
        return '<p>OK</p>'
    return '''
        <h1>Upload new File</h1>
        <form method="post" enctype="multipart/form-data">
        <input type="file" name="file">
        <input type="submit">
        </form>
    '''

if __name__ == "__main__":
    app.run()
