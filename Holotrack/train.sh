#!/bin/bash

python data_slurm.py $(pwd)/.env myconfigs/dce-mbholo-data-32x32x7.yml
python data_slurm.py $(pwd)/.env myconfigs/dce-mbholo-data-64x64x32.yml
python data_slurm.py $(pwd)/.env myconfigs/dce-mbholo-data-128x128x30.yml

python data_slurm.py $(pwd)/.env myconfigs/dce-simon-data-128x128x30.yml
python data_slurm.py $(pwd)/.env myconfigs/dce-simon-data-256x256x50.yml
python data_slurm.py $(pwd)/.env myconfigs/dce-simon-data-512x512x100.yml
python data_slurm.py $(pwd)/.env myconfigs/dce-simon-data-1024x1024x200.yml


python train_slurm.py $(pwd)/.env myconfigs/dce-mbholonet-32x32x7.yml
python train_slurm.py $(pwd)/.env myconfigs/dce-mbholonet-64x64x32.yml
python train_slurm.py $(pwd)/.env myconfigs/dce-mbholonet-128x128x32.yml

python train_slurm.py $(pwd)/.env myconfigs/dce-mbholonet-cross-entropy-32x32x7.yml
python train_slurm.py $(pwd)/.env myconfigs/dce-mbholonet-cross-entropy-64x64x32.yml
python train_slurm.py $(pwd)/.env myconfigs/dce-mbholonet-cross-entropy-128x128x30.yml

python train_slurm.py $(pwd)/.env myconfigs/dce-mbholonet-dicefocalloss-32x32x7.yml
python train_slurm.py $(pwd)/.env myconfigs/dce-mbholonet-dicefocalloss-64x64x32.yml
python train_slurm.py $(pwd)/.env myconfigs/dce-mbholonet-dicefocalloss-128x128x30.yml
