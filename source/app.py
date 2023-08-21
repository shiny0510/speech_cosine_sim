import uuid
from flask import Flask, flash, request, redirect
import librosa
import numpy as np 
from numpy import dot
from numpy.linalg import norm

UPLOAD_FOLDER = 'files'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def root():
    return app.send_static_file('index.html')

@app.route('/save-record', methods=['POST'])
def save_record():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']

    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    file.save("./files/data.wav")
    fs, data  = librosa.load(f"./files/data.wav")
    native_fs, native_data  = librosa.load("Grit1.wav")

    def cos_sim(A, B):
        return dot(A, B)/(norm(A)*norm(B))

    data = librosa.load("files/data.wav")
    #print(len(data[0]))
    native_data  = librosa.load("Grit1.wav")
    #print(len(native_data[0]))
    native_data= list(native_data[0]) 
    data = list(data[0])
    if len(data) > len(native_data):
        while len(data) > len(native_data):
            native_data.append(0)
    else: 
        while len(data) < len(native_data):
            data.append(0)

    #print(len(data))
    #print(len(native_data))
    doc1 = np.array(data)
    doc2 = np.array(native_data)

    return f'유사점수는: {cos_sim(doc1, doc2)}'

if __name__ == '__main__':
    app.run()
