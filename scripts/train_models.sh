#!/bin/sh

echo "Train all models..."
PSHOME=$(pwd)
cd $PSHOME 
python3 train_c-prob_unet.py --what isic3_style_concat --W 0 --epochs 2
python3 train_prob_unet.py --what isic3_style_concat --W 0 --epochs 2
python3 train_prob_unet.py --what isic3_style_0 --W 0 --epochs 2
python3 train_prob_unet.py --what isic3_style_1 --W 0
python3 train_prob_unet.py --what isic3_style_2 --W 0
python3 train_c-ssn.py --what isic3_style_concat --W 0
python3 train_ssn.py --what isic3_style_concat --W 0
python3 train_ssn.py --what isic3_style_0 --W 0
python3 train_ssn.py --what isic3_style_1 --W 0
python3 train_ssn.py --what isic3_style_2 --W 0
echo "Trained all models on ISIC. Training now on PhC dataset..."

python3 train_c-prob_unet.py --what phc_style_concat --W 0
python3 train_prob_unet.py --what phc_style_concat --W 0
python3 train_prob_unet.py --what phc_style_0 --W 0
python3 train_prob_unet.py --what phc_style_1 --W 0
python3 train_prob_unet.py --what phc_style_2 --W 0
python3 train_c-ssn.py --what phc_style_concat --W 0
python3 train_ssn.py --what phc_style_concat --W 0
python3 train_ssn.py --what phc_style_0 --W 0
python3 train_ssn.py --what phc_style_1 --W 0
python3 train_ssn.py --what phc_style_2 --W 0
echo "Trained all models on PhC. Saved models in ./saved_models/"

cd $PSHOME