import warnings
from secrets import randbelow

import numpy as np
from tensorflow.keras.models import model_from_json
# import config
from utils import create_input_array

warnings.filterwarnings("ignore")

__all__ = ["allocate_encrypt_packet"]


# For every dataPacket allocate NNs and encrypt parallely.
def allocate_encrypt_packet(packet, nets, filename, public_key_f):
    public_key = []
    with open(filename, "wb") as f:
        for bit in packet:
            net = randbelow(2)
            public_key.append(net)
            bit_arr = create_input_array(bit)
            encoded = nets[net].predict(bit_arr)
            np.save(f, encoded)
    np_public_key = np.array(public_key)
    np.save(public_key_f, np_public_key)
    return (f, public_key_f)


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

    encrypted_file, public_key = allocate_encrypt_packet(
        packet, nets, "encrypted_data.npy", "public_key.npy"
    )
