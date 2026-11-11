import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

X, y = load_digits(return_X_y=True)
X = X / 16.0
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

raw = LinearSVC(max_iter=5000).fit(x_train, y_train)
print('Raw pixels accuracy:', raw.score(x_test, y_test))

pipe = Pipeline([
    ('rbm', BernoulliRBM(n_components=100, learning_rate=0.06, n_iter=20, random_state=0)),
    ('svc', LinearSVC(max_iter=5000))
])
pipe.fit(x_train, y_train)
print('RBM features accuracy:', pipe.score(x_test, y_test))

rbm = pipe.named_steps['rbm']
plt.figure(figsize=(8, 8))
for i, comp in enumerate(rbm.components_[:64]):
    plt.subplot(8, 8, i + 1)
    plt.imshow(comp.reshape(8, 8), cmap='gray')
    plt.axis('off')
plt.suptitle('RBM learned filters')
plt.show()
