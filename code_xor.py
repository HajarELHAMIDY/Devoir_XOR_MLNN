
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

rn = np.random.RandomState(0)

X = rn.randn(300, 2)
y = np.array(np.logical_xor(X[:, 0] > 0, X[:, 1] > 0),dtype=int)

epochs = 10000
lr = 0.001


import MLNN
model =MLNN.MultiLayerNN(2,20 , 1, "sigmoid", lr, epochs)

model.fit(X,y)

print("model predicted ",model.predict(X))


resultat = plt.figure(figsize=(10,10))
resultat = plot_decision_regions(X=X, y=y, clf=model, legend=2)
plt.title("XOR Model")
plt.show()
