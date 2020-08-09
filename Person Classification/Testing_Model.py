import cv2
import pywt
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler


faceModel = cv2.CascadeClassifier('face.xml')
eyeModel = cv2.CascadeClassifier('eyes.xml')

#Copied from StackOverFlow
def w2d(img, mode='haar', level=1):
    imArray = img
    # Datatype conversions
    # convert to grayscale
    imArray = cv2.cvtColor(imArray, cv2.COLOR_RGB2GRAY)
    # convert to float

    imArray = np.float32(imArray)
    imArray /= 255;
    # compute coefficients
    coeffs = pywt.wavedec2(imArray, mode, level=level)

    # Process Coefficients
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0;

    # reconstruction
    imArray_H = pywt.waverec2(coeffs_H, mode);
    imArray_H *= 255;
    imArray_H = np.uint8(imArray_H)

    return imArray_H

def cropFace(img):
    x, y, w, h = faceModel.detectMultiScale(img)[0]
    face = img[y: y + h, x:x + w]
    print('Cropping Face...')
    cv2.imwrite('test_images/croppedface.jpg', img)

def processImage(img):
    print('Processing Image...')
    cropFace(img)
    face = cv2.imread('test_images/croppedface.jpg')

    face = cv2.resize(face, (32, 32))
    wf = w2d(face, 'db1', 5)
    wf = cv2.resize(wf, (32, 32))
    res = np.vstack((face.reshape(32 * 32 * 3, 1), wf.reshape(32 * 32, 1)))

    return res

def testPerson(img):
    print('Testing image')
    img = processImage(img)
    sc = StandardScaler()
    inputImage  = sc.fit_transform(img)
    inputImage = img.reshape((1, 4096)).astype(float)
    person = {
        0: 'Tom Cruise',
        1: 'Kriti Sanon',
        2: 'Virat Kohli'
    }
    with open('Person_Classification_SVM_MODEL_JOblib.pkl', 'rb') as f:
        model = joblib.load(f)
    output = model.predict(inputImage)

    print()
    return person[output[0]]

imagepath = ''
img = cv2.imread(imagepath)
print('Reading Image...')
person = testPerson(img)
print(person)

cv2.imshow(person, img)
cv2.waitKey(0)
cv2.destroyAllWindows()