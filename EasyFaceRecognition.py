# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 2021

@author: littleHurt
"""

# import necessary package
import cv2
import os
import pickle
import mtcnn
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import Normalizer
from scipy.spatial.distance import cosine
from scipy.sparse import csr_matrix, lil_matrix, coo_matrix, hstack, vstack
from keras.models import load_model




# set basic function
#-----------------------------------------------------------------------------
def get_encode(face_encoder, face, size):
    face = normalize(face)
    face = cv2.resize(face, size)
    encode = face_encoder.predict(np.expand_dims(face, axis = 0,))[0]
    return encode


def get_face(img, box):
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)


l2_normalizer = Normalizer('l2')


def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std


def plt_show(cv_img):
    img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.show()


def load_pickle(path):
    with open(path, 'rb') as f:
        encoding_dict = pickle.load(f)
    return encoding_dict


def save_pickle(path, obj):
    with open(path, 'wb') as file:
        pickle.dump(obj, file)


def read_vc(vc, func_to_call, break_print = ':(', show = False, win_name = '', break_key = 'q', **kwargs):
    while vc.isOpened():
        ret, frame = vc.read()
        if not ret:
            print(break_print)
            break
        res = func_to_call(frame, **kwargs)
        if res is not None:
            frame = res

        if show:
            cv2.imshow(win_name, frame)
        if cv2.waitKey(1) & 0xff == ord(break_key):
            break
# End
#-----------------------------------------------------------------------------






# build function of face detection and recognition
#-----------------------------------------------------------------------------
encoder_model = 'facenet_keras.h5'
required_size = (160, 160) # default size
face_detector = mtcnn.MTCNN()            # set face detector
face_encoder = load_model(encoder_model)


# build a function to generate an embedding from single face
def get_embedding(photo):  
    '''
    photo: path of photo file, only support jpg and png format
    '''
    # get features of face from photo
    img = cv2.imread(photo)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detector.detect_faces(img_rgb)
    
    # convert the features of face into embedding of model
    embeddings = []
    for res in results:
        face, pt_1, pt_2 = get_face(img_rgb, res['box'])
        encode = get_encode(face_encoder, face, required_size)
        encode = l2_normalizer.transform(np.expand_dims(encode, axis = 0))[0]        
        embeddings.append(encode)

    return embeddings    
    # End of funtion
# End
#-----------------------------------------------------------------------------






# compute the cosine distance
#-----------------------------------------------------------------------------

# use the above function to get face embedding
# photo_1 and photo_2 are paths of photo files 
face_1 = get_embedding(photo_1)
face_2 = get_embedding(photo_2)

# compute similarity distance between two photo 
distance.cosine(face_1, face_2)


