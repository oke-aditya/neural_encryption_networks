import numpy as np
from sklearn.metrics import accuracy_score

__all__ = ["create_labels", "create_input_array", "change_output", "accuracy"]


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
    """ Given the predicted array, it compares it with the hashmap and gives the accuracy score """
    Y_pred_int = change_output(Y_pred)
    print(
        "Accuracy for the given batch is :",
        accuracy_score(Y_pred, Y_pred_int) * 100,
        " % ",
    )
