
import librosa
import numpy as np 
from numpy import dot
from numpy.linalg import norm

def cos_sim(A, B):
  return dot(A, B)/(norm(A)*norm(B))

data = librosa.load("files/data.wav")
print(len(data[0]))

native_data  = librosa.load("Grit1.wav")
print(len(native_data[0]))

native_data= list(native_data[0]) 
data = list(data[0])

if len(data) > len(native_data):
    while len(data) > len(native_data):
        native_data.append(0)
else: 
    while len(data) < len(native_data):
        data.append(0)

print(len(data))
print(len(native_data))
doc1 = np.array(data)
doc2 = np.array(native_data)

print(cos_sim(doc1, doc2))