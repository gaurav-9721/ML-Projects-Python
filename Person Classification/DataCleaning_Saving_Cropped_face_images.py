import numpy as np
import cv2 as cv
import os
import seaborn as sb
import matplotlib.pyplot as plt


faceModel = cv.CascadeClassifier('face.xml')
eyeModel = cv.CascadeClassifier('eyes.xml')


def saveCroppedFaces(person, savLoc):
    images = []
    for filename in os.listdir(person):
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            images.append(filename)
    i = 0
    for image in images:

        path = person + '\\' + image
        img = cv.imread(path, 0)
        faces = faceModel.detectMultiScale(img)


        for x, y, h, w in faces:
            crop = img[y:y + h, x:x + w]
            eyes = eyeModel.detectMultiScale(crop)

            if len(eyes) == 2:
                cv.imwrite(savLoc + '\\' + str(i) + '.jpg', crop)
                print("Saving Cropped face from " + image + " in " + str(i) +'.jpg')
                i += 1

#
saveCroppedFaces('Virat Kohli', 'VK_crop')
saveCroppedFaces('Kriti Sanon', 'KS_crop')
saveCroppedFaces('Tom Cruise', 'TC_crop')

