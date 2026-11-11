import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

text = ("to be or not to be that is the question whether tis nobler in the mind to suffer "
        "the slings and arrows of outrageous fortune or to take arms against a sea of troubles "
        "and by opposing end them to die to sleep no more and by a sleep to say we end "
        "the heartache and the thousand natural shocks that flesh is heir to ") * 8
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
seq_len = 40

x_train = np.array([[stoi[c] for c in text[i:i + seq_len]] for i in range(len(text) - seq_len)])
y_train = np.array([stoi[text[i + seq_len]] for i in range(len(text) - seq_len)])

model = models.Sequential([
    layers.Input(shape=(seq_len,)),
    layers.Embedding(len(chars), 64),
    layers.LSTM(128),
    layers.Dense(len(chars), activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(x_train, y_train, epochs=30, batch_size=64, verbose=2)

seed = text[:seq_len]
for _ in range(300):
    x = np.array([[stoi[c] for c in seed[-seq_len:]]])
    p = model.predict(x, verbose=0)[0].astype('float64')
    p /= p.sum()
    seed += itos[np.random.choice(len(chars), p=p)]
print(seed)
