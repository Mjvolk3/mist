---
id: 0a43jc12no3zbp1usbp7xhr
title: MV-m1-startup
desc: ''
updated: 1682705778272
created: 1681385831673
---


## Downloading on M1

```bash
conda env create -f environment-mv-m1.yml
conda activate ms-gen
python -m pip install --force-reinstall -r requirements-mv-m1.txt
python setup.py develop
```

## Sirius Install

```bash
wget https://bio.informatik.uni-jena.de/repository/dist-release-local/de/unijena/bioinf/ms/sirius/4.9.3/sirius-4.9.3-osx64-headless.zip
unzip sirius-4.9.3-osx64-headless.zip
rm sirius-4.9.3-osx64-headless.zip
```

- Can redefine Sirius path where necessary. This is necessary for the quick start `source quickstart/02_run_sirius.sh`

## Quick Start

```python
conda activate ms-gen
source quickstart/00_download_models.sh
python quickstart/01_reformat_mgf.py
source quickstart/02_run_sirius.sh
python quickstart/03_summarize_sirius.py
python quickstart/04_create_lookup.py
source quickstart/05_run_models.sh
```

- All run.

## Training Models

- Download the data to your local machine. This is for **canopus_train**.

```bash
cd data/paired_spectra/
wget https://www.dropbox.com/s/8jn6sz0o3srmtev/canopus_train_public.tar
tar -xvf canopus_train_public.tar
rm canopus_train_public.tar
cd ../../
```

## MIST fingerprint model

```bash
mkdir results/model_train_demos
```

```bash
python run_scripts/train_mist.py --cache-featurizers --dataset-name 'canopus_train_public' --fp-names morgan4096 --num-workers 0 --seed 1 --gpus 0 --split-file 'data/paired_spectra/canopus_train_public/splits/canopus_hplus_100_0.csv' --splitter-name 'preset' --augment-data --augment-prob 0.5 --batch-size 128 --inten-prob 0.1 --remove-prob 0.5 --remove-weights 'exp' --iterative-preds 'growing' --iterative-loss-weight 0.4 --learning-rate 0.00077 --weight-decay 1e-07 --max-epochs 600 --min-lr 0.0001 --lr-decay-time 10000 --lr-decay-frac 0.95 --hidden-size 256 --num-heads 8 --pairwise-featurization --peak-attn-layers 2 --refine-layers 4 --set-pooling 'cls' --spectra-dropout 0.1 --single-form-encoder --recycle-form-encoder --use-cls --cls-type 'ms1' --loss-fn 'cosine' --magma-aux-loss --frag-fps-loss-lambda 8 --magma-modulo 512 --patience 30 --save-dir 'mist_fp_model' --save-dir results/model_train_demos/mist_fp_model
```

- Changed --num-workers 0
  - There is some issue related to multiprocessing on Mac OS
- [x] Model  runs

## FFN binned model

```bash
python run_scripts/train_ffn_binned.py --cache-featurizers --dataset-name 'canopus_train_public' --fp-names morgan4096 --num-workers 10 --seed 1 --gpus 0 --split-file 'data/paired_spectra/canopus_train_public/splits/canopus_hplus_100_0.csv' --splitter-name 'preset' --augment-prob 0.5 --batch-size 128 --inten-prob 0.1 --remove-prob 0.5 --remove-weights 'exp' --iterative-loss-weight 0.5 --iterative-preds 'none' --learning-rate 0.00087 --weight-decay 1e-07 --max-epochs 600 --min-lr 1e-05 --lr-decay-time 10000 --hidden-size 512 --num-spec-layers 2 --num-bins 11000 --spectra-dropout 0.3 --patience 60 --loss-fn 'cosine' --save-dir 'ffn_fp_model' --save-dir results/model_train_demos/ffn_fp_model
```

- Changed --num-workers 10
- [x] Model runs
