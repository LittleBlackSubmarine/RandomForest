import cv2 as cv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


data = pd.read_csv('/home/wolfie/Desktop/Završni/RFA-py/breast_cancer_dataset.csv', dtype=None, delimiter=',')
samples = data.iloc[:, [2, 32]].values
response = data.iloc[:, 1].values


le = LabelEncoder()
response = le.fit_transform(response)

response = np.int32(response)
samples = np.float32(samples)


samples_train, samples_test, response_train, response_test = train_test_split(samples, response, test_size=0.2)


classifier = cv.ml.RTrees_create()

classifier.setActiveVarCount(50)


classifier.train(samples_train, cv.ml.ROW_SAMPLE, response_train)


results = classifier.predict(samples_test, response_test)


#print(results)
#response_test = le.inverse_transform(response_test)

print(response_test)

results = np.asarray(results)

print(results)

#print(accuracy_score(response_test, np.asarray(results)))

