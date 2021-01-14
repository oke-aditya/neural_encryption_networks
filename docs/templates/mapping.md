## Mapping

In order to one hot vectorize the data we need to use mapping.

The hash maps generated have been done using the following algorithm.

```
1. Store all the possible writable characters in a list
characters = [chr(i) for i in range(32, 123)]

2. Generate a random integer between a big range to be assigned

3. For 100 iterations:
    Picking out a random integer from a large range(around 2 power 18)
    Appending the binary representation of the random integer
    End for

4. Shuffle all the sampled numbers that were picked.

5. Assign individual mapping of the bits to the corresponding alphabets in a
random order

```

Code for above pseudocode is as follows

```
def hash_map_generator():ters
    # Take all UTF-8 supported charac
    characters = [chr(i) for i in range(32, 123)]
    # characters.append(' ')
    hash_map_values = []

    # Generate the random integer
    for i in range(100):
        a = random.randint(2 ** 18, 2 ** 19)
        hash_map_values.append(int(bin(a)[2:]))

    # print(hash_map_values)
    
    # Shuffle the mapping.
    random.shuffle(characters)
    random.shuffle(hash_map_values)
    hash_map = dict(zip(characters, hash_map_values))
    # print(hash_map)

    return hash_map
```

The aboce code is located [here](https://github.com/oke-aditya/neural_encryption_networks/tree/master/neural_encryption_networks/src/hash_map_generator.py)
After creating Mapping we can now Train Auto Encoders.
