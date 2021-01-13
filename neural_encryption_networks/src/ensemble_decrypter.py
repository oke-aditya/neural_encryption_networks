import warnings

import numpy as np
from tensorflow.keras.models import model_from_json
# import config
from utils import change_output

warnings.filterwarnings("ignore")

__all__ = ["change_output", "decrypt_file", "decrypt_oh"]

# Model save path
SAVE_PATH = "../../models/"

# Actual models go here !
ENC_MODEL = SAVE_PATH + "encrypter_large.h5"
ENC_JSON = SAVE_PATH + "encrypter_large.json"

DEC_MODEL = SAVE_PATH + "decrypter_large.h5"
DEC_JSON = SAVE_PATH + "decrypter_large.json"


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
    json_file = open("/content/decrypter_large.json", "r")
    read_model_json = json_file.read()
    json_file.close()

    decrypter_large = model_from_json(read_model_json)
    # load weights into new model
    decrypter_large.load_weights("/content/decrypter_large.h5")
    print("Loaded model from disk")

    # load json and create model
    json_file = open("/content/decrypter_small.json", "r")
    read_model_json = json_file.read()
    json_file.close()

    decrypter_small = model_from_json(read_model_json)
    # load weights into new model
    decrypter_small.load_weights("/content/decrypter_small.h5")
    print("Loaded model from disk")

    nets = [decrypter_small, decrypter_large]

    decrypted = decrypt_file("encrypted_data.npy", "public_key.npy")
    print(decrypted)
