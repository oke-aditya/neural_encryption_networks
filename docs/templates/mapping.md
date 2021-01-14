## Mapping

In order to one hot vectorize the data we need to use mapping.

The hash maps generated have been done using the following algorithm.

```
1. Store all the possible writable characters in a list
characters = [chr(i) for i in range(32, 123)]

2. Generate a random integer between a big range to be assigned

3. For 100 iterations:
    Picking out a random integer from a large range(around 2 power 55)
    Appending the binary representation of the random integer
    End for

4. Shuffle all the sampled numbers that were picked.

5. Assign individual mapping of the bits to the corresponding alphabets in a
random order

```

```