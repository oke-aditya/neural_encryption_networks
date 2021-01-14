# Autoencoders

A feed forward autoencoder is a Neural Network containing two feedforward networks.
One is encoder and other is decoder.

![Simple Autoencoder](assets/autoencoder.png)


The encoder is a feedforward network which converts the given input into a latent representation.
The decoder decodes the latent representation to reconstruct the orignal data.
Both are feed-forward networks which contain several non linear layers.
These decrease the total input representation dimension to latent representation dimensions.
## Latent Representation

Latent Representation is a joint representation learnt commonly by both encoder and decoder.
This is unique to encoder as well as decoder.
Learning this latent representation allows us to uniquely encrypt data.

## Training Autoencoder

To trian an autoencoder we should simulatanously train the encoder as well as the decoder.
While the encoder tries to minimize the given input representation, the decoder tries to recreate encoder inputs.
In this process they both learn the latent representation.

## A Small Hint of Supervision

The latent representation can be completely random. Since, it is jointly learnt to minize the loss.
We can add a small hint of supervision by passing labels for encoders and decoders.
This makes the latent distribution a bit consistent. This helps in deterministic execution of AutoEncoder.

