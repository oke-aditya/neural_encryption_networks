from keras.models import model_from_json
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def change_output(arr):
    """' Rounding off the confident outputs """
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


def decrypt_file(file, net_list):
    net_list = np.load(net_list)
    dec_count = 0
    with open(file, "rb") as f:
        tst = ""
        while dec_count < len(net_list):
            arr = np.load(f)
            # print(arr)
            # print(net_list[dec_count])
            out = nets[net_list[dec_count]].predict(arr)
            out = change_output(out)
            # print(out)
            tst += decrypt_oh(out)
            # print(net_list[dec_count])
            # out = nets
            # decrypted = decrypt_net.predict(arr)
            # print(decrypt_out)
            # print(decrypted)
            dec_count += 1
    # print(tst)
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
    json_file = open("/content/decrypter_v1_2_3.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_19bit_decrypter = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_19bit_decrypter.load_weights("/content/decrypter_v1_2_3.h5")
    print("Loaded model from disk")

    # load json and create model
    json_file = open("/content/decrypter_v1_2_1.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_32bit_decrypter = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_32bit_decrypter.load_weights("/content/decrypter_v1_2_1.h5")
    print("Loaded model from disk")

    nets = [loaded_32bit_decrypter, loaded_19bit_decrypter]

    # net_list = np.load('net_list.npy')
    # print(net_list)

    # dec_count = 0
    # with open('multisave.npy','rb') as f:
    #     tst = ''
    #     while(dec_count < len(net_list)):
    #         arr = np.load(f)
    #         # print(arr)
    #         # print(net_list[dec_count])
    #         out = nets[net_list[dec_count]].predict(arr)
    #         out = change_output(out)
    #         # print(out)
    #         tst += decrypt_oh(out)
    #     # print(net_list[dec_count])
    #     # out = nets
    #     # decrypted = decrypt_net.predict(arr)
    #     # print(decrypt_out)
    #     # print(decrypted)
    #         dec_count += 1
    # print(tst)

    decrypted = decrypt_file("/content/multisave.npy", "/content/net_list.npy")
    print(decrypted)
