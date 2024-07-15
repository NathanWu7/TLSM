# Tactile latent space mapping (TLSM)
This repository provides support for the IROS paper: "RTTF: Rapid Tactile Transfer Framework". This section implements the tactile latent space mapping.

## Install
```bash
pip install -r requirements.txt
```

## Usage
1. Data preparation
```bash
mkdir data/tatcile
mkdir data/tactile_pair
```
for the data of senser 0 <br>
put unpaired data in data/tactile/tactile0 <br>
put paired data in data/tactile_pair/tactile0 <br>

2. Tactile VAE training
```bash
python3 train_vae.py
```

3. Tactile LSM training
```bash
python3 train_mapping.py
```
