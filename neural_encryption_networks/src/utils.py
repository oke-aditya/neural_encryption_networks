import random

import numpy as np
from sklearn.metrics import accuracy_score

__all__ = [
    "create_labels",
    "create_input_array",
    "convert",
    "change_output",
    "accuracy",
    "read_from_file",
    "sequence",
    "random_paragraph_generator",
    "generate_hashmap"
]


def generate_hashmap(limit_min, limit_max):
    characters = [chr(i) for i in range(32, 123)]
    # characters.append(' ')
    hash_map_values = []

    for i in range(100):
        a = random.randint(limit_min, limit_max)
        hash_map_values.append(int(bin(a)[2:]))

    # print(hash_map_values)

    random.shuffle(characters)
    random.shuffle(hash_map_values)
    hash_map = dict(zip(characters, hash_map_values))
    # print(hash_map)

    return hash_map


def read_from_file(file):
    m = ""
    with open(file, "r") as f:
        m += f.readline()
        m += " "
    m.rstrip(" ")
    return m


def sequence(word_length):
    characters = [chr(i) for i in range(32, 123)]
    random.shuffle(characters)
    word = "".join(characters[:word_length])
    return word


def random_paragraph_generator():
    no_of_words = random.randint(1000, 1500)
    para = ""
    for i in range(no_of_words):
        word_length = random.randint(1, 10)
        word = sequence(word_length)
        para += word
        para += " "
    # para = para[2:len(para)-2]
    return para


def convert(paragraph, hashmap):
    output = []
    for i in paragraph:
        output.append(list(map(int, str(hashmap[i]))))
    return output


def create_labels(paragraph, hashmap):
    # paragraph = paragraph.lower()
    output = []
    for i in paragraph:
        output.append(list(map(int, str(hashmap[i]))))
    return np.array(output)


def create_input_array(word):
    enc_l = []
    for i in word:
        arr = np.zeros(91)
        enc = ord(i) - 32
        arr[enc] += 1
        enc_l.append(arr)
    return np.array(enc_l)


def change_output(arr):
    row = arr.shape[0]
    col = arr.shape[1]
    for i in range(row):
        for j in range(col):
            if arr[i][j] > 0.5:
                arr[i][j] = 1
            else:
                arr[i][j] = 0
    arr.astype("int")
    return arr


def accuracy(Y_pred, Y_train):
    Y_pred_int = change_output(Y_pred)
    print(
        "Accuracy for the given batch is :",
        accuracy_score(Y_pred, Y_pred_int) * 100,
        " % ",
    )
