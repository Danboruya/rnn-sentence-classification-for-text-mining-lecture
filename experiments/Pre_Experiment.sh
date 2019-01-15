#!/usr/bin/env bash

# pre-experiment
python ../app.py --cell_type="GRU" --n_epoch=400 --dropout_keep_prob=0.7 --exp_name="pre_exp";