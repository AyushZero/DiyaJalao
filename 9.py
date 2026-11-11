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
