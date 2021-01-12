import warnings
from secrets import randbelow

import numpy as np
from keras.models import model_from_json

warnings.filterwarnings("ignore")

__all__ = ["create_input_array", "allocate_encrypt_packet"]


def create_input_array(word):
    enc_l = []
    for i in word:
        arr = np.zeros(91)
        enc = ord(i) - 32
        arr[enc] += 1
        enc_l.append(arr)
    return np.array(enc_l)


# For every dataPacket allocate NNs and encrypt parallely.
def allocate_encrypt_packet(packet, nets, filename, net_list_f):
    net_list = []
    # name_list = []
    with open(filename, "wb") as f:
        for bit in packet:
            net = randbelow(2)
            net_list.append(net)
            bit_arr = create_input_array(bit)
            encoded = nets[net].predict(bit_arr)
            # print(encoded)
            np.save(f, encoded)
    # np.savez("try1",encoded)

    # name_list.append()
    # encoded = encoded.tolist()
    # encoded_data.append(encoded)
    np_net_list = np.array(net_list)
    np.save(net_list_f, np_net_list)
    return (f, net_list_f)


if __name__ == "__main__":

    # load json and create model
    json_file = open("/content/encrypter_small.json", "r")
    read_model_json = json_file.read()
    json_file.close()

    encrypter_small = model_from_json(read_model_json)
    # load weights into new model
    encrypter_small.load_weights("/content/encrypter_small.h5")
    print("Loaded model from disk")

    # load json and create model
    json_file = open("/content/encrypter_large.json", "r")
    read_model_json = json_file.read()
    json_file.close()

    encrypter_large = model_from_json(read_model_json)
    # load weights into new model
    encrypter_large.load_weights("/content/encrypter_large.h5")
    print("Loaded model from disk")

    encrypter_small.compile(
        optimizer="adam", loss="mean_squared_error", metrics=["acc"]
    )

    encrypter_large.compile(
        optimizer="adam", loss="mean_squared_error", metrics=["acc"]
    )

    packet = "abcd"
    nets = [encrypter_small, encrypter_large]

    encoded_file, net_list = allocate_encrypt_packet(packet, nets, "multisave.npy", "net_list.npy")
