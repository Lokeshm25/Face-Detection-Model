#!pip install mtcnn
#!pip install keras-facenet
import cv2 as cv
import tensorflow as tf
import os
import numpy as np
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

img = cv.imread("/content/drive/MyDrive/Colab Notebooks/dataset/lokesh_maheshwari/9.jpg")
plt.imshow(img)

img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.imshow(img)

detector = MTCNN()
faces = detector.detect_faces(img)

faces

x,y,w,h = faces[0]['box']
img = cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 5)
plt.imshow(img)

my_face = img[y:y+h, x:x+w]
#plt.imshow(my_face)
my_face = cv.resize(my_face, (160,160))
plt.imshow(my_face)

my_face



class FACELOADING:
    def __init__(self, directory):
        self.directory = directory
        self.target_size = (160,160)
        self.X = []
        self.Y = []
        self.detector = MTCNN()


    def extract_face(self, filename):
        img = cv.imread(filename)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        x,y,w,h = self.detector.detect_faces(img)[0]['box']
        x,y = abs(x), abs(y)
        face = img[y:y+h, x:x+w]
        face_arr = cv.resize(face, self.target_size)
        return face_arr


    def load_faces(self, dir):
        FACES = []
        for im_name in os.listdir(dir):
            try:
                path = dir + im_name
                single_face = self.extract_face(path)
                FACES.append(single_face)
            except Exception as e:
                pass
        return FACES

    def load_classes(self):
        for sub_dir in os.listdir(self.directory):
            path = self.directory +'/'+ sub_dir+'/'
            FACES = self.load_faces(path)
            labels = [sub_dir for _ in range(len(FACES))]
            print(f"Loaded successfully: {len(labels)}")
            self.X.extend(FACES)
            self.Y.extend(labels)

        return np.asarray(self.X), np.asarray(self.Y)


    def plot_images(self):
        plt.figure(figsize=(18,16))
        for num,image in enumerate(self.X):
            ncols = 3
            nrows = len(self.Y)//ncols + 1
            plt.subplot(nrows,ncols,num+1)
            plt.imshow(image)
            plt.axis('off')

faceloading = FACELOADING("/content/drive/MyDrive/Colab Notebooks/dataset")
X, Y = faceloading.load_classes()

plt.figure(figsize=(16,12))
for num,image in enumerate(X):
    ncols = 3
    nrows = len(Y)//ncols + 1
    plt.subplot(nrows,ncols,num+1)
    plt.imshow(image)
    plt.axis('off')

from keras_facenet import FaceNet
embedder = FaceNet()

def get_embedding(face_img):
    face_img = face_img.astype('float32') # 3D(160x160x3)
    face_img = np.expand_dims(face_img, axis=0)
    # 4D (Nonex160x160x3)
    yhat= embedder.embeddings(face_img)
    return yhat[0] # 512D image (1x1x512)

EMBEDDED_X = []

for img in X:
    EMBEDDED_X.append(get_embedding(img))

EMBEDDED_X = np.asarray(EMBEDDED_X)

np.savez_compressed('faces_embeddings_done_4classes.npz', EMBEDDED_X, Y)

Y

from sklearn.preprocessing import LabelEncoder
import pickle
encoder = LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)
encoder_path = "/content/drive/MyDrive/Colab Notebooks/label_encoder.pkl"
with open(encoder_path, 'wb') as f:
    pickle.dump(encoder, f)

print(f"✅ LabelEncoder saved at: {encoder_path}")

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(EMBEDDED_X, Y, shuffle=True, random_state=17)

from sklearn.svm import SVC
model = SVC(kernel='linear', probability=True)
model.fit(X_train, Y_train)

ypreds_train = model.predict(X_train)
ypreds_test = model.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy_score(Y_train, ypreds_train)

accuracy_score(Y_test,ypreds_test)

t_im = cv.imread("/content/drive/MyDrive/Colab Notebooks/test.jpg")
t_im = cv.cvtColor(t_im, cv.COLOR_BGR2RGB)
x,y,w,h = detector.detect_faces(t_im)[0]['box']

t_im = t_im[y:y+h, x:x+w]
t_im = cv.resize(t_im, (160,160))
test_im = get_embedding(t_im)

test_im = [test_im]
ypreds = model.predict(test_im)

ypreds

encoder.inverse_transform(ypreds)


#save the model
drive_path = "/content/drive/MyDrive/Colab Notebooks/svm_model_160x160.pkl"
with open(drive_path,'wb') as f:
    pickle.dump(model,f)
print(f"Model saved at: {drive_path}")