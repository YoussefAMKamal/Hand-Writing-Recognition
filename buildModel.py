import numpy
import random
import keras
from keras.datasets import mnist
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def load_dataset(): 
    (training_image, training_label), (test_image, test_label) = mnist.load_data()

    training_image = training_image.reshape((training_image.shape[0], 28, 28, 1))
    test_image = test_image.reshape((test_image.shape[0], 28, 28, 1))

    training_image = training_image.astype('float32')/255
    test_image = test_image.astype('float32')/255

    training_image, val_image, training_label, val_label = train_test_split(training_image, training_label, test_size=0.2)

    return training_image, training_label, test_image, test_label, val_image, val_label

def build_model():
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), name='Convolutional_layer_1'))
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', name='Convolutional_layer_2'))
    model.add(keras.layers.MaxPooling2D((2, 2), name='Maxpooling_2D_1'))
    model.add(keras.layers.Dropout(0.35, name='Droupout_1'))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', name='Convolutional_layer_3'))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', name='Convolutional_layer_4'))
    model.add(keras.layers.MaxPooling2D((2, 2), name='Maxpooling_2D_2'))
    model.add(keras.layers.Dropout(0.35, name='Droupout_2'))
    model.add(keras.layers.Flatten(name='Flatten'))
    model.add(keras.layers.Dense(512, activation='relu', name='Hidden_layer1'))
    model.add(keras.layers.Dense(256, activation='relu', name='Hidden_layer2'))
    model.add(keras.layers.Dense(128, activation='relu', name='Hidden_layer3'))
    model.add(keras.layers.Dense(10, activation='softmax', name='Output_layer'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def fit_model():
    training_image, training_label, test_image, test_label, val_image, val_label = load_dataset()

    model = build_model()

    Old = model.fit(training_image, training_label, epochs=10, batch_size = 32, validation_data=(val_image, val_label))

    print("\nCalculating Accuracy and Loss in Training and Testing:\n")

    training_loss, training_accuracy = model.evaluate(training_image, training_label)

    print('Training Accuracy {}%'.format(round(float(training_accuracy)*100, 4)))
    print('Training Loss {}%'.format(round(float(training_loss)*100, 4)))

    test_loss, test_accuracy = model.evaluate(test_image, test_label)

    model.save('model.keras')

    print('Test Accuracy {}%'.format(round(float(test_accuracy)*100, 4)))
    print('Test Loss {}%'.format(round(float(test_loss)*100, 4)))

    plt.plot(Old.history['accuracy'], label='accuracy')
    plt.plot(Old.history['loss'], label='loss')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='upper right')
    plt.show()

    print("Predecting...")
    predictions = numpy.argmax(model.predict(test_image),axis = -1)

    plt.figure(figsize=(10, 10))
    x = random.sample(range(len(test_label)), 16)
    for i in range(16):
        plt.subplot(4,4,i+1)
        plt.imshow(test_image[x[i]].reshape(28,28,1),cmap='grey')
        plt.title("Predicted {}, Class {}".format(predictions[x[i]], test_label[x[i]]))
        plt.tight_layout()
    plt.show()