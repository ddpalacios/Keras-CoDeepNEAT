
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class Setup:

    def dataset(self, data):
         # # The data, split between train and test sets:
        (x_train, y_train), (x_test, y_test) = data
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')
        print("Loaded mnist data")

        num_classes = 10

        # # Convert class vectors to binary class matrices.
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train = x_train.reshape(60000, 28, 28, 1)
        x_test = x_test.reshape(10000, 28, 28, 1)
        x_train /= 255
        x_test /= 255
        validation_split = 0.15
        #training
        batch_size = 128
        datagen = self._augment(x_train)

        return x_train,y_train,x_test,y_test,validation_split, batch_size, datagen


    def compiler_(self, loss= "categorical_crossentropy", optimizer= "keras.optimizers.Adam", lr=0.005, metrics = "accuracy"):
        
        optimizer = "{}({})".format(optimizer, lr)
        compiler = {"loss": loss, "optimizer":optimizer, "metrics":[metrics]}

        return compiler



    def _augment(self, data):
         # #data augmentation
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            )

        datagen.fit(data)
        return datagen
