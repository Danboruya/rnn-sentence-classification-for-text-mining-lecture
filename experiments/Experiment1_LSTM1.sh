#!/usr/bin/env bash

# LSTM experiment 1 [Learning rate]
python ../app.py --cell_type="LSTM" --n_epoch=400 --learning_rate=1e-4 --dropout_keep_prob=0.7 --exp_name="exp01_1e-4";
python ../app.py --cell_type="LSTM" --n_epoch=400 --learning_rate=1e-5 --dropout_keep_prob=0.7 --exp_name="exp01_1e-5";
python ../app.py --cell_type="LSTM" --n_epoch=400 --learning_rate=1e-6 --dropout_keep_prob=0.7 --exp_name="exp01_1e-6";

# LSTM experiment 2 [Layer]
python ../app.py --cell_type="LSTM" --n_epoch=400 --learning_rate=1e-4 --n_layer=1 --dropout_keep_prob=0.7 --exp_name="exp02_1layer";
python ../app.py --cell_type="LSTM" --n_epoch=400 --learning_rate=1e-4 --n_layer=2 --dropout_keep_prob=0.7 --exp_name="exp02_2layer";
python ../app.py --cell_type="LSTM" --n_epoch=400 --learning_rate=1e-4 --n_layer=3 --dropout_keep_prob=0.7 --exp_name="exp02_3layer";
python ../app.py --cell_type="LSTM" --n_epoch=400 --learning_rate=1e-4 --n_layer=4 --dropout_keep_prob=0.7 --exp_name="exp02_4layer";

echo "Script has been done. [LSTM-1]"
