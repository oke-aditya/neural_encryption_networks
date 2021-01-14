import warnings

import config
import numpy as np
from tensorflow.keras import Sequential, layers, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import model_from_json
from utils import (accuracy, change_output, create_input_array, create_labels,
                   generate_hashmap, random_paragraph_generator)

warnings.filterwarnings("ignore")


if __name__ == "__main__":

    hashmap = generate_hashmap(2 ** 55, 2 ** 56)
    # print(len(hashmap))

    l = []
    for i in hashmap.keys():
        l.append(i)

    # print(sorted(l))

    train_string = random_paragraph_generator()
    print(train_string)

    test_string = """ abcdefghigjklmnopqrstuvx ysx d go abcdef hidsog """

    for i in train_string:
        if ord(i) < 32 or ord(i) > 122:
            print(i, ord(i), chr(ord(i)), train_string.find(i))
            # print(ord(i))

    for i in test_string:
        if ord(i) < 32 or ord(i) > 122:
            print(i, ord(i), chr(ord(i)), train_string.find(i))
            # print(ord(i)

    X_train = create_input_array(train_string)
    print(X_train.shape)

    X_test = create_input_array(test_string)
    print(X_test.shape)

    print(X_train[0])

    Y_train = create_labels(train_string, hashmap)
    print(Y_train.shape)

    Y_test = create_labels(test_string, hashmap)
    print(Y_test.shape)

    # print(Y_train)
    print(Y_train.shape)

    # print(Y_test)
    print(Y_test.shape)

    encrypter = Sequential()
    encrypter.add(layers.Dense(91, input_shape=(91,)))
    encrypter.add(layers.LeakyReLU())

    encrypter.add(layers.Dense(82))
    encrypter.add(layers.LeakyReLU())

    encrypter.add(layers.Dense(74))
    encrypter.add(layers.LeakyReLU())

    encrypter.add(layers.Dense(68))
    encrypter.add(layers.LeakyReLU())

    encrypter.add(layers.Dense(64))
    encrypter.add(layers.LeakyReLU())

    encrypter.add(layers.Dense(56))
    encrypter.add(layers.LeakyReLU())

    encrypter.add(layers.Dense(56))
    encrypter.add(layers.LeakyReLU())

    learning_rate = config.ENC_LEARNING_RATE
    epochs = config.ENC_EPOCHS
    batch_size = config.ENC_BATCH_SIZE

    optim = optimizers.Adam(lr=config.ENC_LEARNING_RATE)

    checkpoint = ModelCheckpoint(
        config.ENC_LARGE_CHK,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
    )
    reducelr = ReduceLROnPlateau(
        monitor="val_loss", verbose=1, patience=5, factor=0.05, min_lr=0.003
    )

    encrypter.compile(optimizer=optim, loss="mean_squared_error", metrics=["acc"])

    encrypter.summary()

    history = encrypter.fit(
        X_train,
        Y_train,
        batch_size=config.ENC_BATCH_SIZE,
        epochs=config.ENC_EPOCHS,
        verbose=1,
        callbacks=[checkpoint, reducelr],
        validation_split=config.ENC_VALIDATION_SPLIT,
    )

    output = encrypter.predict(X_train)

    print(output[0])

    print(Y_train[0])

    print(output.shape)

    Y_pred = encrypter.predict(X_train)
    accuracy(Y_pred, Y_train)

    print(Y_train.shape)

    Y_test_pred = encrypter.predict(X_test)
    accuracy(Y_test_pred, Y_test)

    # serialize model to JSON
    model_json = encrypter.to_json()
    with open(config.ENC_LARGE_JSON, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    encrypter.save_weights(config.ENC_LARGE_MODEL)
    print("Saved model to disk")

    # load json and create model
    json_file = open(config.ENC_LARGE_JSON, "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(config.ENC_LARGE_MODEL)
    print("Loaded model from disk")

    print("Encrypter can be loaded and run")

    decrypter = Sequential()

    decrypter.add(layers.Dense(56, input_shape=(56,)))
    decrypter.add(layers.LeakyReLU())

    decrypter.add(layers.Dense(64))
    decrypter.add(layers.LeakyReLU())

    decrypter.add(layers.Dense(72))
    decrypter.add(layers.LeakyReLU())

    decrypter.add(layers.Dense(80))
    decrypter.add(layers.LeakyReLU())

    decrypter.add(layers.Dense(85))
    decrypter.add(layers.LeakyReLU())

    decrypter.add(layers.Dense(88))
    decrypter.add(layers.LeakyReLU())

    decrypter.add(layers.Dense(91))
    decrypter.add(layers.LeakyReLU())

    decrypter.add(layers.Dense(91))
    decrypter.add(layers.LeakyReLU())

    learning_rate = config.DEC_LEARNING_RATE
    epochs = config.DEC_EPOCHS
    batch_size = config.DEC_BATCH_SIZE

    decrypter_optimizer = optimizers.Adam(lr=learning_rate)

    decrypter.compile(
        optimizer=decrypter_optimizer, loss="mean_squared_error", metrics=["acc"]
    )

    decrypter_X_train = Y_train
    decrypter_Y_train = X_train

    decrypter_X_test = Y_test_pred
    decrypter_Y_test = X_test

    print(decrypter_X_train.shape)
    print(decrypter_Y_train.shape)

    print(decrypter_X_test.shape)
    print(decrypter_Y_test.shape)

    checkpoint = ModelCheckpoint(
        config.DEC_LARGE_CHK,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
    )
    reducelr = ReduceLROnPlateau(
        monitor="val_loss", verbose=1, patience=5, factor=0.05, min_lr=0.003
    )

    decrypter_history = decrypter.fit(
        decrypter_X_train,
        decrypter_Y_train,
        batch_size=config.DEC_BATCH_SIZE,
        epochs=config.DEC_EPOCHS,
        verbose=1,
        callbacks=[checkpoint, reducelr],
        validation_split=config.DEC_VALIDATION_SPLIT,
    )

    decrypted_text = decrypter.predict(decrypter_X_train)

    decrypted_int_text = change_output(decrypted_text)

    print(decrypted_text[0])

    print(decrypted_int_text[0])

    print(decrypter_Y_train[0])

    print(np.argmax(decrypted_text[0]))

    print(np.argmax(decrypter_Y_train[0]))

    print(decrypted_text[1])

    print(decrypted_int_text[1])

    print(decrypter_Y_train[1])

    print(np.argmax(decrypted_text[1]))

    print(np.argmax(decrypter_Y_train[1]))

    accuracy(decrypted_int_text, decrypter_Y_train)

    decrypted_Y_test_pred = decrypter.predict(decrypter_X_test)

    decrypted_Y_test_pred = change_output(decrypted_Y_test_pred)

    accuracy(decrypted_Y_test_pred, decrypter_Y_test)

    # serialize model to JSON
    model_json = decrypter.to_json()
    with open(config.DEC_LARGE_JSON, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    decrypter.save_weights(config.DEC_LARGE_MODEL)
    print("Saved model to disk")

    # load json and create model
    json_file = open(config.DEC_LARGE_JSON, "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(config.DEC_LARGE_MODEL)
    print("Loaded model from disk")

    print("Decrypter can be loaded and run")
