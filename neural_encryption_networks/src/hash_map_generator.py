import random

__all__ = ["hash_map_generator"]


def hash_map_generator():
    characters = [chr(i) for i in range(32, 123)]
    # characters.append(' ')
    hash_map_values = []

    for i in range(100):
        a = random.randint(2 ** 18, 2 ** 19)
        hash_map_values.append(int(bin(a)[2:]))

    # print(hash_map_values)

    random.shuffle(characters)
    random.shuffle(hash_map_values)
    hash_map = dict(zip(characters, hash_map_values))
    # print(hash_map)

    return hash_map
