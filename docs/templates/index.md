# ICADCML 2021 A Novel Approach to Encrypt Data using Deep Neural Networks

In this paper we propose a novel approach to Encrpyt data using Deep Neural Networks.
We propose an Autoencoder techinque which can sucessfully encrypt and decrypt data.
We secure this method using keys by ensembling autoencoders.

## Project layout

This is how the project is structured.
We also provide Colab Notebooks that can be used to reproduce our results.

```

├── docs                            # Documentation files built using mkdocs
├── models                          # Models that we trained
├── neural_encryption_networks      # This folder contains all code.
│   ├── __init__.py
│   ├── notebooks                   # Reproducible notebooks.
│   └── src                         # Python scripts.
└── requirements.txt                # To install stuff.
```

## Colab Notebooks

We provide Colab notebooks to directly play with. The are available in `neural_encryption_networks/notebooks` folder too.

- [Encrypting with Deep Neural Networks](https://colab.research.google.com/drive/1E6sgBR0NLuUjqRD-705Bf_oG9kEQP1uF?usp=sharing)

- [Ensemble Network for Keys](https://colab.research.google.com/drive/1YRdKXOPMcpeLH7s1IOzSVmd_eBdFWKZy?usp=sharing)


## Runing locally

Install the requirments by running

```
pip install -r requirements.txt
```

The code uses Tensorflow 2.4. And is Tested on Python 3.6+

We recommend using virtual environements using Conda or similar to avoid conflicts.

```
cd neural_encryption_networks/src
```

This folder contains all the code you need !

## Citation

We will provide a Bibtex entry soon.
For now people can cite this GitHub repository.
