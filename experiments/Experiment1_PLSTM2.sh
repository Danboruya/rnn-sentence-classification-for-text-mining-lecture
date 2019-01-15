#!/usr/bin/env bash

# PLSTM experiment 3 [Cell]
python ../app.py --cell_type="PLSTM" --n_epoch=400 --learning_rate=1e-4 --dropout_keep_prob=0.7 --n_cell=16 --exp_name="exp03_16cell";
python ../app.py --cell_type="PLSTM" --n_epoch=400 --learning_rate=1e-4 --dropout_keep_prob=0.7 --n_cell=32 --exp_name="exp03_32cell";
python ../app.py --cell_type="PLSTM" --n_epoch=400 --learning_rate=1e-4 --dropout_keep_prob=0.7 --n_cell=64 --exp_name="exp03_64cell";
python ../app.py --cell_type="PLSTM" --n_epoch=400 --learning_rate=1e-4 --dropout_keep_prob=0.7 --n_cell=128 --exp_name="exp03_128cell";

# PLSTM experiment 4 [Embedding dimension]
python ../app.py --cell_type="PLSTM" --n_epoch=400 --learning_rate=1e-4 --dropout_keep_prob=0.7 --embedding_dim=32 --exp_name="exp04_32dim";
python ../app.py --cell_type="PLSTM" --n_epoch=400 --learning_rate=1e-4 --dropout_keep_prob=0.7 --embedding_dim=64 --exp_name="exp04_64dim";
python ../app.py --cell_type="PLSTM" --n_epoch=400 --learning_rate=1e-4 --dropout_keep_prob=0.7 --embedding_dim=128 --exp_name="exp04_128dim";

echo "Script has been done. [PLSTM-2]"
