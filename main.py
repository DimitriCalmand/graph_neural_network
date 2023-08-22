import numpy as np
import numpy_graph as npg
import layer
from dense import *
import activations
from model import Sequential
def loss(y_pred, y_true):
    res = -(y_true * npg.log(y_pred) + (1 - y_true) * npg.log(1 - y_pred))
    res = res.mean(axis=0)
    return res

model = Sequential()
model.add(Dense(3, (2,)))
model.add(activations.sigmoid())
model.add(Dense(1))
model.add(activations.sigmoid())
X_train = np.array([[0,0],[0,1],[1,0],[1,1]])
y_train = np.array([[0],[1],[1],[0]])
X_train = npg.g_array(X_train)
y_train = npg.g_array(y_train)
model.compile(loss)
model.fit(X_train, y_train, 1000, 4, verbose = 100)

