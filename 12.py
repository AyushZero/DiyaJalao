import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report

rng = np.random.default_rng(0)
pers = ['john', 'mary', 'ravi', 'anita', 'david', 'sara']
locs = ['chennai', 'paris', 'london', 'delhi', 'tokyo', 'berlin']
fill = ['lives', 'in', 'works', 'at', 'visited', 'and', 'met', 'the', 'went', 'to', 'with']
tag_names = ['O', 'B-PER', 'B-LOC']
vocab = ['<pad>'] + pers + locs + fill
w2i = {w: i for i, w in enumerate(vocab)}
maxlen = 12

X, Y = [], []
for _ in range(3000):
    n = rng.integers(6, maxlen + 1)
    words, tags = [], []
    for _ in range(n):
        r = rng.random()
        if r < 0.2:
            words.append(pers[rng.integers(len(pers))])
            tags.append(1)
        elif r < 0.4:
            words.append(locs[rng.integers(len(locs))])
            tags.append(2)
        else:
            words.append(fill[rng.integers(len(fill))])
            tags.append(0)
    X.append([w2i[w] for w in words] + [0] * (maxlen - n))
    Y.append(tags + [0] * (maxlen - n))
X, Y = np.array(X), np.array(Y)
x_train, y_train, x_test, y_test = X[:2500], Y[:2500], X[2500:], Y[2500:]

model = models.Sequential([
    layers.Input(shape=(maxlen,)),
    layers.Embedding(len(vocab), 64, mask_zero=True),
    layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
    layers.TimeDistributed(layers.Dense(len(tag_names), activation='softmax'))
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=8, batch_size=64, validation_split=0.1)

pred = model.predict(x_test, verbose=0).argmax(-1)
mask = x_test != 0
print(classification_report(y_test[mask], pred[mask], target_names=tag_names))
