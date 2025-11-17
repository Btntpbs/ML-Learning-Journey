## TASK 2.1 ## 
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

plt.figure(figsize=(9, 9))
for i in range(81):
    plt.subplot(9, 9, i + 1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(str(y_train[i]), fontsize=8)0
    plt.axis('off')
plt.tight_layout()
plt.show()

## TASK 2.2 ##
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)
x_test  = np.expand_dims(x_test, -1)

y_train = keras.utils.to_categorical(y_train, 10)
y_test  = keras.utils.to_categorical(y_test, 10)

model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=5,
          batch_size=128,
          validation_split=0.1,
          verbose=2)


test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")


pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
true = np.argmax(y_test, axis=1)
cm = tf.math.confusion_matrix(labels=true, predictions=pred)
print("Confusion matrix:\n", cm.numpy())

## TASK 2.3 ## 
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0


num_classes = 10
y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_test_cat  = keras.utils.to_categorical(y_test,  num_classes)
x_train_cnn = np.expand_dims(x_train, -1)
x_test_cnn  = np.expand_dims(x_test,  -1)


def add_salt_pepper_noise(images, amount=0.15):
    noisy = images.copy()
    H, W = images.shape[1], images.shape[2]
    num_noisy = int(amount * H * W)
    for i in range(images.shape[0]):

        r = np.random.randint(0, H, num_noisy)
        c = np.random.randint(0, W, num_noisy)
        noisy[i, r, c] = 1.0

        r = np.random.randint(0, H, num_noisy)
        c = np.random.randint(0, W, num_noisy)
        noisy[i, r, c] = 0.0
    return noisy

x_train_noisy = add_salt_pepper_noise(x_train, amount=0.15)
x_test_noisy  = add_salt_pepper_noise(x_test,  amount=0.15)
x_train_noisy_cnn = np.expand_dims(x_train_noisy, -1)
x_test_noisy_cnn  = np.expand_dims(x_test_noisy,  -1)


def build_cnn():
    model = keras.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model_clean = build_cnn()
model_clean.fit(
    x_train_cnn, y_train_cat,
    epochs=5,
    batch_size=128,
    validation_split=0.1,
    verbose=2
)

print("\n=== Noise Evaluation ===")
clean_loss, clean_acc = model_clean.evaluate(x_test_cnn, y_test_cat, verbose=0)
print(f"Accuracy on CLEAN test images : {clean_acc:.4f}")

noisy_loss, noisy_acc = model_clean.evaluate(x_test_noisy_cnn, y_test_cat, verbose=0)
print(f"Accuracy on NOISY test images (15% salt & pepper): {noisy_acc:.4f}")


plt.figure(figsize=(9, 3))
for i in range(9):
    plt.subplot(1, 9, i + 1)
    plt.imshow(x_test_noisy[i], cmap='gray')
    plt.axis('off')
plt.suptitle("MNIST Test Samples with 15% Salt & Pepper Noise")
plt.tight_layout()
plt.show()
