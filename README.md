# ICADCML 2021 Paper A Novel Approach to Encrypt Data using Deep Neural Networks

<div align="center">

![Check Code formatting](https://github.com/oke-aditya/neural_encryption_networks/workflows/Check%20Code%20formatting/badge.svg)
![Build mkdocs](https://github.com/oke-aditya/template_python/workflows/Build%20mkdocs/badge.svg)
![Deploy mkdocs](https://github.com/oke-aditya/template_python/workflows/Deploy%20mkdocs/badge.svg)

</div>

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

## Documentation

For detailed documentation visit [here](https://oke-aditya.github.io/neural_encryption_networks/)

## Colab Notebooks

We provide Colab notebooks to directly play with. The are available in `neural_encryption_networks/notebooks` folder too.

- [Encrypter Small]()

- [Encrypter Large]()

- [Ensemble Network]()



## Runing locally

Install the requirments by running

```
pip install -r requirements.txt
```

The code uses Tensorflow 2.4.

We recommend using virtual environements using Conda or similar to avoid conflicts.

```
cd neural_encryption_networks/src
```

This folder contains all the code you need !

## Citation

We will provide a Bibtex entry soon.
For now people can cite this GitHub repository.
