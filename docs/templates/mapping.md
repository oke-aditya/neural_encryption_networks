## Mapping

In order for the one hot vector generation to happen, it was essential to use some sort of mapping, and thus, *hash maps* were chosen for this purpose.

One requirement for this task was that the data structure being used should be able gives us back the value in the least possible time. Currently our dataset of characters being encoded is limited to the English alphabets, but if this is extended to characters of other langauges, the data structure becomes an important consideration as well during the training of the network. So to make sure that our future objectives can be easily extended without much changes, we chose hashmaps, which take up a bit of space, but the lookups are considerably faster, especially for larger number of objects.

The hash maps generated have been done using the **random number** generator in python, and the number of digits being generated for each mapping, is equivalent to the number of bits the neural network will be trained on.

Now for the model demonstration purposes, we used the `random` module in python for all the values that we generated. For better security, and enterprise level security, as well as by recommendation of python docs, `secret` package could be used for the same purpose.
