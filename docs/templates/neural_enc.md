## Encrypting Data Using Auto-Encoders

## Training Set and Mapping

### Enocder As Encrypter

After we have created mapping, we can now Encrypt data using Deep Neural Networks.
We create 2 simple Feed Forward networks an encoder and decoder.

The encoder acts as encrypter while the decoder acts as decrypter.

A simple encoder using Tensorflow.Keras API.

```
encrypter = Sequential()
encrypter.add(layers.Dense(91, input_shape=(91,)))
encrypter.add(layers.LeakyReLU())

encrypter.add(layers.Dense(72))
encrypter.add(layers.LeakyReLU())

encrypter.add(layers.Dense(64))
encrypter.add(layers.LeakyReLU())

encrypter.add(layers.Dense(48))
encrypter.add(layers.LeakyReLU())

encrypter.add(layers.Dense(36))
encrypter.add(layers.LeakyReLU())

encrypter.add(layers.Dense(32))
encrypter.add(layers.LeakyReLU())

encrypter.add(layers.Dense(32))
encrypter.add(layers.LeakyReLU())
```

### Decoder As Decrypter


```
decrypter = Sequential()

decrypter.add(layers.Dense(32, input_shape=(32,)))
decrypter.add(layers.LeakyReLU())

decrypter.add(layers.Dense(40))
decrypter.add(layers.LeakyReLU())

decrypter.add(layers.Dense(46))
decrypter.add(layers.LeakyReLU())

decrypter.add(layers.Dense(54))
decrypter.add(layers.LeakyReLU())

decrypter.add(layers.Dense(64))
decrypter.add(layers.LeakyReLU())

decrypter.add(layers.Dense(91))
decrypter.add(layers.LeakyReLU())

decrypter.add(layers.Dense(91))
decrypter.add(layers.LeakyReLU())

decrypter.add(layers.Dense(91))
decrypter.add(layers.LeakyReLU())
```

## Training

We train Encoder and Decoder with `Mean Squared Error` Loss, `Adam` Optimizer.

We keep a small learning rate around `1e-3`.

Since there are not many gradients to compute, `batch size` is kept 1.

Training both Encoder and Decoder Jointly takes around 3-4 mins over CPU and 2 mins over GPU.

## Ensembling Networks

Once we have trained one set of enocder (encrypter) and decoder (decrypter).
We can use similar configuration and train another.
We trained 2 such set of encrypter and decrypters.
Both had slightly different mapping, created by mapping algorithm.
Encrypter, Decrypter small network had hashmap with encoding size 32 and a larger with encoding size 56.
This allows us to create secure networks, by ensembling them.

## Inference With Ensemble Networks


