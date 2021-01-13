import warnings

import numpy as np
from tensorflow.keras.models import model_from_json

import config
from utils import change_output

warnings.filterwarnings("ignore")

__all__ = ["change_output", "decrypt_file", "decrypt_oh"]


def decrypt_file(file, net_list):
    net_list = np.load(net_list)
    dec_count = 0
    with open(file, "rb") as f:
        tst = ""
        while dec_count < len(net_list):
            arr = np.load(f)
            out = nets[net_list[dec_count]].predict(arr)
            out = change_output(out)
            tst += decrypt_oh(out)
            dec_count += 1
    return tst


def decrypt_oh(oh_dec):
    tst = ""
    for i in oh_dec:
        pos = int((np.where(i == 1)[0][0]))
        pos += 32
        tst += chr(pos)
    return tst


if __name__ == "__main__":

    # load json and create model
    json_file = open(config.DEC_LARGE_JSON, "r")
    read_model_json = json_file.read()
    json_file.close()

    decrypter_large = model_from_json(read_model_json)
    # load weights into new model
    decrypter_large.load_weights(config.DEC_LARGE_MODEL)
    print("Loaded model from disk")

    # load json and create model
    json_file = open(config.DEC_SMALL_JSON, "r")
    read_model_json = json_file.read()
    json_file.close()

    decrypter_small = model_from_json(read_model_json)
    # load weights into new model
    decrypter_small.load_weights(config.DEC_SMALL_MODEL)
    print("Loaded model from disk")

    nets = [decrypter_small, decrypter_large]

    decrypted = decrypt_file(config.ENCRYPTED_FILE_PATH, config.PUBLIC_KEY_PATH)
    print(decrypted)
