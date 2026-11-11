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
