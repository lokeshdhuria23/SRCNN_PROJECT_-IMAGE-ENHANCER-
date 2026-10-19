#!/bin/bash
echo "Setting up SRCNN on Remote GPU..."

# 1. Install dependencies
echo "Installing Python requirements..."
pip install -r requirements.txt

# 2. Download and extract dataset (DIV2K High-Res)
echo "Downloading DIV2K dataset..."
mkdir -p data/DIV2K_train_HR
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip -O DIV2K_train_HR.zip

echo "Extracting dataset..."
unzip -q DIV2K_train_HR.zip -d data/DIV2K_train_HR/
rm DIV2K_train_HR.zip

# 3. Run training
echo "Starting training..."
python train.py
