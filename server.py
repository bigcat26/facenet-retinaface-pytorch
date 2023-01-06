import cv2
import json
import math
import numpy as np
from utils import utils

from retinaface import Retinaface
from flask import Flask, request, Response
# from werkzeug.utils import secure_filename
from facedb import FaceDatabase

db = FaceDatabase()
app = Flask(__name__)
retinaface = Retinaface()

def distance(embeddings1, embeddings2, distance_metric=0):
    if distance_metric==0:
        # Euclidian distance
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 0)
    elif distance_metric==1:
        # Distance based on cosine similarity
        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=0)
        norm = np.linalg.norm(embeddings1, axis=0) * np.linalg.norm(embeddings2, axis=0)
        similarity = dot / norm
        dist = np.arccos(similarity) / math.pi
    else:
        raise 'Undefined distance metric %d' % distance_metric 
        
    return dist

@app.route("/unreg", methods = ['GET'])
def unregister():
    if request.args['pid']:
        db.unregister(request.args['pid'])
        result = {'status': 'OK'}
        return Response(json.dumps(result), mimetype='text/json')

@app.route('/reg', methods = ['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        fp = request.files['file'].read()
        buf = np.frombuffer(fp, np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        landmarks = retinaface.face_detect(img)
        if len(landmarks) != 1:
            result = {'status': 'Error', 'message': f'invalid face count: {len(landmarks)}'}
            return Response(json.dumps(result), mimetype='text/json')
        landmarks = landmarks.squeeze()
        npimg = np.asarray(img, np.uint8)
        face    = npimg[int(landmarks[1]):int(landmarks[3]), int(landmarks[0]):int(landmarks[2])]
        face, _ = utils.align_face_5kp(face, landmarks[5:].reshape((5, 2)))
        feat    = retinaface.extract_feature(face)
        pid     = db.register(name, feat)
        result = {'status': 'OK', 'name': name, 'id': pid}
        return Response(json.dumps(result, ensure_ascii=False), mimetype='text/json')
    return '''
        <h1>Register</h1>
        <form method="post" enctype="multipart/form-data">
            <label for="name">Name</label><input type="text" id="name" name="name"/><br/>
            <label for="file">Image</label><input type="file" id="file" name="file"/><br/>
            <input type="submit"/><br/>
        </form>
    '''

def feature_match(feat, threshold = 0.9):
    pid = -1
    distance = 999999999.9
    features = db.features()
    for f in features:
        dist = utils.face_distance(f.feat, feat, 0)
        if dist < distance:
            distance = dist
            pid = f.pid
    return pid if distance < threshold else -1, distance

@app.route('/match', methods = ['GET', 'POST'])
def match():
    if request.method == 'POST':
        fp = request.files['file'].read()
        buf = np.frombuffer(fp, np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        npimg = np.asarray(img, np.uint8)
        landmarks = retinaface.face_detect(img)
        output = []
        for landmark in landmarks:
            face    = npimg[int(landmark[1]):int(landmark[3]), int(landmark[0]):int(landmark[2])]
            face, _ = utils.align_face_5kp(face, landmark[5:].reshape((5, 2)))
            feat    = retinaface.extract_feature(face)
            pid, dist = feature_match(feat)
            output.append({
                'left': int(landmark[0]),
                'top': int(landmark[1]),
                'width': int(landmark[2] - landmark[0]),
                'height': int(landmark[3] - landmark[1]),
                'confidence': float(landmark[4]),
                'match': int(pid),
                'distance': float(dist),
                'keypoints': [
                    [int(landmark[5]), int(landmark[6])],
                    [int(landmark[7]), int(landmark[8])],
                    [int(landmark[9]), int(landmark[10])],
                    [int(landmark[11]), int(landmark[12])],
                    [int(landmark[13]), int(landmark[14])]
                ]
            })
        result = {'status': 'OK', 'data': output}
        return Response(json.dumps(result, ensure_ascii=False), mimetype='text/json')
    return '''
        <h1>Match</h1>
        <form method="post" enctype="multipart/form-data">
            <label for="file">Image</label><input type="file" id="file" name="file"/><br/>
            <input type="submit"/><br/>
        </form>
    '''

def get_feature_from_image(img):
    landmarks = retinaface.face_detect(img)
    if len(landmarks) != 1:
        return False, None
    landmarks = landmarks.squeeze()
    npimg = np.asarray(img, np.uint8)
    face    = npimg[int(landmarks[1]):int(landmarks[3]), int(landmarks[0]):int(landmarks[2])]
    face, _ = utils.align_face_5kp(face, landmarks[5:].reshape((5, 2)))
    return True, retinaface.extract_feature(face)

@app.route('/feature', methods = ['GET', 'POST'])
def feature():
    if request.method == 'POST':
        fp = request.files['file'].read()
        buf = np.frombuffer(fp, np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        ok, feat = get_feature_from_image(img)
        if not ok:
            return Response(json.dumps({'status': 'Error', 'message': f'extract feat from image failed'}), mimetype='text/json')

        result = {'status': 'OK', 'data': feat.tolist()}
        return Response(json.dumps(result), mimetype='text/json')
    return '''
        <h1>Extract Feature</h1>
        <form method="post" enctype="multipart/form-data">
        <input type="file" name="file">
        <input type="submit">
        </form>
    '''

@app.route('/diff', methods = ['GET', 'POST'])
def diff():
    if request.method == 'POST':
        fp = request.files['file1'].read()
        buf = np.frombuffer(fp, np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        ok, feat1 = get_feature_from_image(img)
        if not ok:
            return Response(json.dumps({'status': 'Error', 'message': f'extract feat from image 1 failed'}), mimetype='text/json')

        fp = request.files['file2'].read()
        buf = np.frombuffer(fp, np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        ok, feat2 = get_feature_from_image(img)
        if not ok:
            return Response(json.dumps({'status': 'Error', 'message': f'extract feat from image 2 failed'}), mimetype='text/json')

        euc_dist = distance(feat1, feat2, 0)
        cos_dist = distance(feat1, feat2, 1)
        old_dist = utils.face_distance(feat1, feat2, 0)

        result = {'status': 'OK', 'euc_dist': float(euc_dist), 'old_dist': float(old_dist), 'cos_dist': float(cos_dist)}
        return Response(json.dumps(result), mimetype='text/json')
    return '''
        <h1>Compare face</h1>
        <form method="post" enctype="multipart/form-data">
        <input type="file" name="file1"/><br/>
        <input type="file" name="file2"/><br/>
        <input type="submit"><br/>
        </form>
    '''

@app.route('/detect', methods = ['GET', 'POST'])
def detect():
    if request.method == 'POST':
        fp = request.files['file'].read()
        npimg = np.frombuffer(fp, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        landmarks = retinaface.face_detect(img)
        output = []
        for landmark in landmarks:
            output.append({
                'left': int(landmark[0]),
                'top': int(landmark[1]),
                'width': int(landmark[2] - landmark[0]),
                'height': int(landmark[3] - landmark[1]),
                'confidence': float(landmark[4]),
                'keypoints': [
                    [int(landmark[5]), int(landmark[6])],
                    [int(landmark[7]), int(landmark[8])],
                    [int(landmark[9]), int(landmark[10])],
                    [int(landmark[11]), int(landmark[12])],
                    [int(landmark[13]), int(landmark[14])]
                ]
            })
        result = {'status': 'OK', 'data': output}
        return Response(json.dumps(result), mimetype='text/json')
    return '''
        <h1>Face Detect</h1>
        <form method="post" enctype="multipart/form-data">
        <input type="file" name="file">
        <input type="submit">
        </form>
    '''

@app.route('/clip', methods = ['GET', 'POST'])
def clip():
    if request.method == 'POST':
        fp = request.files['file'].read()
        buf = np.frombuffer(fp, np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        landmarks = retinaface.face_detect(img)
        if len(landmarks) != 1:
            result = {'status': 'Error', 'message': f'invalid face count: {len(landmarks)}'}
            return Response(json.dumps(result), mimetype='text/json')
        landmarks = landmarks.squeeze()
        npimg = np.asarray(img, np.uint8)
        face    = npimg[int(landmarks[1]):int(landmarks[3]), int(landmarks[0]):int(landmarks[2])]
        face, _ = utils.align_face_5kp(face, landmarks[5:].reshape((5, 2)))
        ret, data = cv2.imencode('.png', face)
        if not ret:
            result = {'status': 'Error', 'message': f'encode image error'}
            return Response(json.dumps(result), mimetype='text/json')
        return Response(data.tobytes(), mimetype='image/png')
    return '''
        <h1>Clip</h1>
        <form method="post" enctype="multipart/form-data">
            <label for="file">Image</label><input type="file" id="file" name="file"/><br/>
            <input type="submit"/><br/>
        </form>
    '''

if __name__ == "__main__":
    with open('credentials.json', 'r', encoding='utf8') as f:
        cfg = json.load(f)
    db.open(**cfg)
    app.run(debug=True)
