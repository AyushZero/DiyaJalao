# Experiment 8: LeNet for image classification
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(6, 5, activation='tanh', padding='same'),
    layers.AveragePooling2D(),
    layers.Conv2D(16, 5, activation='tanh'),
    layers.AveragePooling2D(),
    layers.Flatten(),
    layers.Dense(120, activation='tanh'),
    layers.Dense(84, activation='tanh'),
    layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=10, validation_split=0.1)
print('Test acc:', model.evaluate(x_test, y_test)[1])

plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Experiment 9: ResNet for detecting objects
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, y_train = x_train[:5000], y_train[:5000]
x_test, y_test = x_test[:1000], y_test[:1000]

def prep(x, y):
    x = tf.image.resize(tf.cast(x, tf.float32), (224, 224))
    return preprocess_input(x), y

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).map(prep).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).map(prep).batch(32)

base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base.trainable = False
model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_ds, validation_data=test_ds, epochs=3)
print('Test acc:', model.evaluate(test_ds)[1])

plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Experiment 10: Transfer learning using VGG16
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
tr = np.isin(y_train.ravel(), [3, 5])
te = np.isin(y_test.ravel(), [3, 5])
x_train, y_train = x_train[tr][:2000], (y_train[tr][:2000] == 5).astype('int32')
x_test, y_test = x_test[te][:500], (y_test[te][:500] == 5).astype('int32')

def prep(x, y):
    x = tf.image.resize(tf.cast(x, tf.float32), (224, 224))
    return preprocess_input(x), y

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).map(prep).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).map(prep).batch(32)

base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base.trainable = False
model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(2, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_ds, validation_data=test_ds, epochs=5)

base.trainable = True
for layer in base.layers[:-4]:
    layer.trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history2 = model.fit(train_ds, validation_data=test_ds, epochs=2)
print('Test acc:', model.evaluate(test_ds)[1])

plt.plot(history.history['accuracy'] + history2.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'] + history2.history['val_accuracy'], label='val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Experiment 11: Character level language modeling with RNN
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

# Experiment 12: LSTM network for Named Entity Recognition
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

# Experiment 13: Neural Machine Translation with attention
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

# Experiment 14: Scene understanding using RBMs
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

# Experiment 15: Generating images with GAN
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

(x, _), _ = tf.keras.datasets.mnist.load_data()
x = (x.reshape(-1, 28, 28, 1).astype('float32') - 127.5) / 127.5
ds = tf.data.Dataset.from_tensor_slices(x).shuffle(60000).batch(256)

latent = 100
G = models.Sequential([
    layers.Input(shape=(latent,)),
    layers.Dense(7 * 7 * 128, use_bias=False),
    layers.Reshape((7, 7, 128)),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.Conv2DTranspose(64, 4, strides=2, padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.Conv2DTranspose(1, 4, strides=2, padding='same', activation='tanh')
])
D = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(64, 4, strides=2, padding='same'),
    layers.LeakyReLU(0.2),
    layers.Conv2D(128, 4, strides=2, padding='same'),
    layers.LeakyReLU(0.2),
    layers.Flatten(),
    layers.Dense(1)
])

bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
g_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
d_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

@tf.function
def train_step(real):
    z = tf.random.normal([tf.shape(real)[0], latent])
    with tf.GradientTape() as gt, tf.GradientTape() as dt:
        fake = G(z, training=True)
        r = D(real, training=True)
        f = D(fake, training=True)
        g_loss = bce(tf.ones_like(f), f)
        d_loss = bce(tf.ones_like(r), r) + bce(tf.zeros_like(f), f)
    g_opt.apply_gradients(zip(gt.gradient(g_loss, G.trainable_variables), G.trainable_variables))
    d_opt.apply_gradients(zip(dt.gradient(d_loss, D.trainable_variables), D.trainable_variables))
    return g_loss, d_loss

g_hist, d_hist = [], []
for epoch in range(5):
    for batch in ds:
        g_loss, d_loss = train_step(batch)
    g_hist.append(float(g_loss))
    d_hist.append(float(d_loss))
    print('Epoch', epoch + 1, 'G loss', float(g_loss), 'D loss', float(d_loss))

plt.plot(g_hist, label='G loss')
plt.plot(d_hist, label='D loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

samples = G(tf.random.normal([16, latent]), training=False).numpy() * 0.5 + 0.5
plt.figure(figsize=(4, 4))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(samples[i, :, :, 0], cmap='gray')
    plt.axis('off')
plt.show()
