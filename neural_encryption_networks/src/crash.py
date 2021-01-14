import random


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
    return para.encode("utf-8")


if __name__ == "__main__":
    # res = generate_hashmap(2 ** 55, 2 ** 56)
    # print(res)
    para = random_paragraph_generator()
    print(para)
