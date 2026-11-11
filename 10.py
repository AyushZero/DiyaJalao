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
