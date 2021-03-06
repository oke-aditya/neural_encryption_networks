## Mapping to learn Latent Distribution.

To give Small Hint of Supervision we use mapping. 
Mapping helps to create a almost consistent latent distribution, which is learnt by encoder.

To create mapping, we use hashmaps which are generated by following algorithm.

### Pseudocode

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

### Code

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

The above code is located [here](https://github.com/oke-aditya/neural_encryption_networks/tree/master/neural_encryption_networks/src/hash_map_generator.py)

Once we create Mapping we can Train AutoEncoders.
