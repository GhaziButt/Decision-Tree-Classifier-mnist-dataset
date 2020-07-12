import numpy as np
import matplotlib.pyplot as mp
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

import io
import tensorflow as tf
from sklearn import metrics

(X_T , Y_T) , (X_TE , Y_TE) =tf.keras.datasets.mnist.load_data()

clf = DecisionTreeClassifier()
 
X_T = X_T.reshape((60000 , 784)) 
X_TE = X_TE.reshape((10000 ,784))

clf.fit(X_T,Y_T)

y_pred = clf.predict(X_TE)

print("Accuracy using Decision Trees is :" , metrics.accuracy_score(Y_TE ,y_pred)*100)