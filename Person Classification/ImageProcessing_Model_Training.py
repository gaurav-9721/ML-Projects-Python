import numpy as np
import cv2
import pywt
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pandas as pd
import joblib
import seaborn as sb

X = []  #Features or Input Variable
Y = []  #Output or Target Variable


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

def createXYData(person, p):
    images = []
    for filename in os.listdir(person):
        if filename.endswith(".jpg"):
            images.append(filename)
        print('Reading ' + filename + ' of ' + person)
        path = person + '\\' + filename
        img = cv2.imread(path)

        img = cv2.resize(img, (32, 32))
        wf = w2d(img, 'db1', 5)
        wf = cv2.resize(wf, (32, 32))
        res = np.vstack((img.reshape(32 * 32 * 3, 1), wf.reshape(32 * 32, 1)))

        X.append(res)
        Y.append(p)

print()
createXYData('TC_crop', 0)
createXYData('KS_crop', 1)
createXYData('VK_crop', 2)

X = np.array(X).reshape(len(X), len(X[0])).astype(float)

# Training Our Model

xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.25)

training_model = {
    'SVM': {
        'model': SVC(gamma='auto', probability=True),
        'params': {
            'svc__C': [1, 10, 500, 100, 1000],
            'svc__kernel': ['linear', 'rbf']
        }
    }
    # I commented this models but you can use them to test out more.
    # 'Random Forest': {
    #     'model': RandomForestClassifier(),
    #     'params': {
    #         'randomforestclassifier__n_estimators': [1, 5, 10, 20, 40, 50, 100]
    #     }
    # },
    #
    # 'Logistic Regression': {
    #     'model': LogisticRegression(),
    #     'params': {
    #         'logisticregression__C': [1, 5, 10]
    #     }
    # }

}

score = []
model_name = []
model_params = []
models = {}

for name, model in training_model.items():
    pipe = make_pipeline(StandardScaler(), model['model'])
    gcv = GridSearchCV(pipe, model['params'], cv=5, return_train_score=False)
    print('Training Model using '+ name)

    gcv.fit(xtrain, ytrain)

    score.append(gcv.best_score_)
    model_name.append(name)
    model_params.append(gcv.best_params_)
    models[name] = gcv.best_estimator_

#Using GridSearchCV I tested out Logistic Regression, SVM and Random Forests with different parameters.
#We found out that Logistic Regression had maxiumum score out of those three.

#This DataFrame has information about those Model accuracy and best Score.
df = pd.DataFrame(list(zip(model_name, model_params, score)), columns=['Model', 'Parameters', 'Score'])

# Saving Model in a file
joblib.dump(models['SVM'], 'Person_Classification_SVM_MODEL_JOblib')