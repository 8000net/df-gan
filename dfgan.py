from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD
from keras.datasets import mnist
import numpy as np
from PIL import Image
import argparse
import math

from os import listdir
from os.path import join
from scipy.ndimage import imread
from sklearn.metrics import confusion_matrix


def generator_model():
    model = Sequential()
    model.add(Dense(input_dim=100, output_dim=1024))
    model.add(Activation('tanh'))
    model.add(Dense(128*5*5))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((5, 5, 128), input_shape=(128*5*5,)))
    model.add(UpSampling2D(size=(7, 7)))
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(5, 5)))
    model.add(Conv2D(32, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(3, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    return model


def discriminator_model():
    model = Sequential()
    model.add(Conv2D(64, (5, 5),padding='same',input_shape=(350, 350, 3)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(8, 8)))
    model.add(Conv2D(128, (5, 5)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(8, 8)))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model


def generator_containing_discriminator(g, d):
    model = Sequential()
    model.add(g)
    d.trainable = False
    model.add(d)
    return model


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1], 3),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, :]
    return image

def load_data(data_dirs = []):
    X = []

    for data_dir in data_dirs:
        files = listdir(data_dir)
        for fp in files:
            img = imread(join(data_dir, fp))
            X.append(img)

    X = np.array(X)
    np.random.shuffle(X)
    return X



def train(BATCH_SIZE, load_models):
    X_train = load_data(data_dirs = ["df-data/ian-real"])
    X_fake = load_data(data_dirs = ["df-data/ian-ian"])
    X_train = (X_train.astype(np.float32) - 127.5)/127.5
    X_fake = (X_fake.astype(np.float32) - 127.5)/127.5
    d = discriminator_model()
    g = generator_model()
    d_on_g = generator_containing_discriminator(g, d)
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    d_on_g.compile(loss='binary_crossentropy', optimizer=g_optim)
    d.trainable = True
    d.compile(loss='binary_crossentropy', optimizer=d_optim)
    if(load_models):
        d.load_weights('discriminator.h5')
        g.load_weights('generator.h5')
    FAKE_BATCH_SIZE = int(BATCH_SIZE/2)
    for epoch in range(100):
        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0]/BATCH_SIZE))
        for index in range(int(X_train.shape[0]/BATCH_SIZE)):
            noise = np.random.uniform(-1, 1, size=(FAKE_BATCH_SIZE, 100))
            image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            generated_images = g.predict(noise, verbose=0)
            faceswapped_images = X_fake[index*FAKE_BATCH_SIZE:(index+1)*FAKE_BATCH_SIZE]
            if index % 20 == 0:
                image = combine_images(generated_images)
                image = image*127.5+127.5
                Image.fromarray(image.astype(np.uint8)).save(
                    str(epoch)+"_"+str(index)+".png")
                d.save_weights('discriminator.h5')
                g.save_weights('generator.h5')
            print(image_batch.shape, generated_images.shape, faceswapped_images.shape)
            X = np.concatenate((image_batch, generated_images, faceswapped_images))
            y = [1] * BATCH_SIZE + [0] * (2*FAKE_BATCH_SIZE)
            d_loss = d.train_on_batch(X, y)
            print("batch %d d_loss : %f" % (index, d_loss))
            noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
            d.trainable = False
            g_loss = d_on_g.train_on_batch(noise, [1] * BATCH_SIZE)
            d.trainable = True
            print("batch %d g_loss : %f" % (index, g_loss))

        d.save_weights('discriminator.h5')
        g.save_weights('generator.h5')


def generate(BATCH_SIZE):
    g = generator_model()
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    g.load_weights('generator.h5')
    noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
    generated_images = g.predict(noise, verbose=1)
    image = combine_images(generated_images)
    image = image*127.5+127.5
    Image.fromarray(image.astype(np.uint8)).save(
        "generated_image.png")

def evaluate():
    d = discriminator_model()
    d.compile(loss='binary_crossentropy', optimizer="SGD")
    d.load_weights('discriminator.h5')

    X_real = load_data(data_dirs = ["df-data/jake-real"])
    X_fake = load_data(data_dirs = ["df-data/jake-jake"])

    X_real = (X_real.astype(np.float32) - 127.5)/127.5
    X_fake = (X_fake.astype(np.float32) - 127.5)/127.5

    X = np.concatenate((X_real, X_fake))
    y_real = [1] * X_real.shape[0] + [0] * X_fake.shape[0]

    y_pred = d.predict_classes(X)

    print(confusion_matrix(y_real, y_pred))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument('--load_weights', dest='loadweights', action='store_true')
    parser.set_defaults(loadweights=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    print("Keras Implementation of DF-GAN")
    args = get_args()
    if args.mode == "train":
        train(BATCH_SIZE=args.batch_size, args.loadweights)
    elif args.mode == "generate":
        generate(BATCH_SIZE=args.batch_size)
    elif args.mode == "evaluate":
        evaluate()
