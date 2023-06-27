# That Label's Got Style: Handling Label Style Bias for Uncertain Image Segmentation

Kilian Zepf, Eike Petersen, Jes Frellsen, Aasa Feragen, ICLR 2023

[[Paper]]([https://arxiv.org/abs/2103.16265](https://openreview.net/pdf?id=wZ2SVhOTzBX))

This repository contains the code for the experiments presented in the paper as well as a new datasets for studying varying label styles in uncertain image segmentation. The code is based on PyTorch. 

## Dependencies

All dependencies are listed the the file `requirements.txt`. You can set-up a new virtual environment and install dependencies with 

```
pip install -r requirements.txt
```

## Data

Download the datasets by running

```
./scripts/download_data.sh
```

This will downoad the isic and PhC-U373 datasets and unpack them to `./data/`

## Train (conditioned) uncertain segmentation models

To train all 4 models on both datasets (=8 models total), with pre-tuned hyperparameters, execute the script:

```
./scripts/train.sh
```

## Test the models and generate figures of the paper

Lorem Ipsum

