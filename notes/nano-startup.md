---
id: cotxx0j5txxfjevr4qe76rr
title: Nano Startup
desc: ''
updated: 1681393055575
created: 1681384889258
---
## Downloading on Nano

- First thing run `scl enable devtoolset-4 bash` from terminal

```python
conda create -n ms-gen pip  python==3.8
conda activate ms-gen
conda install -c conda-forge matplotlib
pip install rdkit
pip install pytorch-lightning
pip3 install torch torchvision torchaudio
pip install torchmetrics
pip install einops
python -m pip install ipykernel
conda install -c conda-forge nb_conda_kernels
pip install seaborn
pip install h5py
pip install -U "ray[tune]"
pip install umap-learn
pip install optuna
# start of requirements txt
pip install git+https://github.com/LiyuanLucasLiu/RAdam
pip install git+https://github.com/ray-project/ray_lightning#ray_lightning
pip install scikit-learn
pip install pathos
pip install hyperopt
pip install setuptools==59.5.0

# pip install protobuf==3.20.1 # added bc pytorch lightning needs this version... it should go near the beginning to avoid the later errors... This is not needed if the install below won't work anyways.
# pip install -U "ray[air]" This one doesn't work

pip install CairoSVG
```

- After environment setup

## Package Setup

```python
python setup.py develop
```

- Everything seemed to work

## Sirius Install

```bash
wget https://bio.informatik.uni-jena.de/repository/dist-release-local/de/unijena/bioinf/ms/sirius/4.9.3/sirius-4.9.3-linux64-headless.zip
unzip sirius-4.9.3-linux64-headless.zip
rm sirius-4.9.3-linux64-headless.zip
```

## Quick Start

- Start an interactive session on the GPU to see if the environment and the rest of the software is working properly.

```bash
qsub -I -l nodes=nano1:ppn=1:gpus=1,walltime=3600"
```

- Alternatively you can you the **nano interactive** task if you are in VsCode.

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
python run_scripts/train_mist.py --cache-featurizers --dataset-name 'canopus_train_public' --fp-names morgan4096 --num-workers 12 --seed 1 --gpus 1 --split-file 'data/paired_spectra/canopus_train_public/splits/canopus_hplus_100_0.csv' --splitter-name 'preset' --augment-data --augment-prob 0.5 --batch-size 128 --inten-prob 0.1 --remove-prob 0.5 --remove-weights 'exp' --iterative-preds 'growing' --iterative-loss-weight 0.4 --learning-rate 0.00077 --weight-decay 1e-07 --max-epochs 600 --min-lr 0.0001 --lr-decay-time 10000 --lr-decay-frac 0.95 --hidden-size 256 --num-heads 8 --pairwise-featurization --peak-attn-layers 2 --refine-layers 4 --set-pooling 'cls' --spectra-dropout 0.1 --single-form-encoder --recycle-form-encoder --use-cls --cls-type 'ms1' --loss-fn 'cosine' --magma-aux-loss --frag-fps-loss-lambda 8 --magma-modulo 512 --patience 30 --save-dir 'mist_fp_model' --save-dir results/model_train_demos/mist_fp_model
```

- I have change `--gpus 0` to `--gpus 1`
- [x] Model  runs

## FFN binned model

```bash
python run_scripts/train_ffn_binned.py --cache-featurizers --dataset-name 'canopus_train_public' --fp-names morgan4096 --num-workers 12 --seed 1 --gpus 1 --split-file 'data/paired_spectra/canopus_train_public/splits/canopus_hplus_100_0.csv' --splitter-name 'preset' --augment-prob 0.5 --batch-size 128 --inten-prob 0.1 --remove-prob 0.5 --remove-weights 'exp' --iterative-loss-weight 0.5 --iterative-preds 'none' --learning-rate 0.00087 --weight-decay 1e-07 --max-epochs 600 --min-lr 1e-05 --lr-decay-time 10000 --hidden-size 512 --num-spec-layers 2 --num-bins 11000 --spectra-dropout 0.3 --patience 60 --loss-fn 'cosine' --save-dir 'ffn_fp_model' --save-dir results/model_train_demos/ffn_fp_model
```

- I have change `--gpus 0` to `--gpus 1`
