import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

pairs = [("i am happy", "je suis content"), ("you are happy", "tu es content"),
         ("he is sad", "il est triste"), ("she is sad", "elle est triste"),
         ("i am sad", "je suis triste"), ("you are sad", "tu es triste"),
         ("he is happy", "il est content"), ("she is happy", "elle est content"),
         ("we are happy", "nous sommes content"), ("we are sad", "nous sommes triste"),
         ("i am tired", "je suis fatigue"), ("he is tired", "il est fatigue"),
         ("she is tired", "elle est fatigue"), ("you are tired", "tu es fatigue"),
         ("we are tired", "nous sommes fatigue")]

src_words = ['<pad>'] + sorted({w for s, t in pairs for w in s.split()})
tgt_words = ['<pad>', '<sos>', '<eos>'] + sorted({w for s, t in pairs for w in t.split()})
s2i = {w: i for i, w in enumerate(src_words)}
t2i = {w: i for i, w in enumerate(tgt_words)}
i2t = {i: w for w, i in t2i.items()}

enc_x = np.array([[s2i[w] for w in s.split()] for s, t in pairs])
tgt = np.array([[t2i['<sos>']] + [t2i[w] for w in t.split()] + [t2i['<eos>']] for s, t in pairs])
dec_x, dec_y = tgt[:, :-1], tgt[:, 1:]

enc_in = layers.Input(shape=(None,))
enc_emb = layers.Embedding(len(src_words), 64)(enc_in)
enc_out, h, c = layers.LSTM(128, return_sequences=True, return_state=True)(enc_emb)

dec_in = layers.Input(shape=(None,))
dec_emb = layers.Embedding(len(tgt_words), 64)(dec_in)
dec_out = layers.LSTM(128, return_sequences=True)(dec_emb, initial_state=[h, c])

context = layers.Attention()([dec_out, enc_out])
merged = layers.Concatenate()([dec_out, context])
out = layers.Dense(len(tgt_words), activation='softmax')(merged)

model = models.Model([enc_in, dec_in], out)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit([enc_x, dec_x], dec_y, epochs=300, verbose=0)
print('Train acc:', model.evaluate([enc_x, dec_x], dec_y, verbose=0)[1])

def translate(sentence):
    x = np.array([[s2i[w] for w in sentence.split()]])
    dec = [t2i['<sos>']]
    for _ in range(6):
        p = model.predict([x, np.array([dec])], verbose=0)[0, -1]
        nxt = int(p.argmax())
        if nxt == t2i['<eos>']:
            break
        dec.append(nxt)
    return ' '.join(i2t[i] for i in dec[1:])

for s, t in pairs[:6]:
    print(s, '->', translate(s), '| reference:', t)
