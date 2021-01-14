# Keys For Secure Encryption

For secure encryption algorithm one need keys; Namely a public key and a private key.

## Public Key

The public key is the number of encrypters used and their ordering.
This enables our decryptors to know the decryption ordering. The public key
does not risk our encryption algorithm. It does not expose the mapping of encrypters.

## Private Key

The latent distribution mapping is a private key.
It differs in number of encoded bits for each encryter-decrypter set,
as well as the learnt latent distribution varies for each encoder and decoder.

Thus we can create a safe 2 key encrpytion-decryption mechanism.