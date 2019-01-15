#!/usr/bin/env bash

# CNN experiment 3 [Filter size]
python ../app.py --cell_type="CNN" --n_epoch=400 --learning_rate=1e-4 --filter_sizes=3,4,4 --dropout_keep_prob=0.7 --exp_name="exp03_344";
python ../app.py --cell_type="CNN" --n_epoch=400 --learning_rate=1e-4 --filter_sizes=3,5,5 --dropout_keep_prob=0.7 --exp_name="exp03_355";
python ../app.py --cell_type="CNN" --n_epoch=400 --learning_rate=1e-4 --filter_sizes=3,5,6 --dropout_keep_prob=0.7 --exp_name="exp03_356";
python ../app.py --cell_type="CNN" --n_epoch=400 --learning_rate=1e-4 --filter_sizes=3,6,6 --dropout_keep_prob=0.7 --exp_name="exp03_366";

# CNN experiment 4 [Embedding dimension]
python ../app.py --cell_type="CNN" --n_epoch=400 --learning_rate=1e-4 --dropout_keep_prob=0.7 --embedding_dim=32 --exp_name="exp04_32dim";
python ../app.py --cell_type="CNN" --n_epoch=400 --learning_rate=1e-4 --dropout_keep_prob=0.7 --embedding_dim=64 --exp_name="exp04_64dim";
python ../app.py --cell_type="CNN" --n_epoch=400 --learning_rate=1e-4 --dropout_keep_prob=0.7 --embedding_dim=128 --exp_name="exp04_128dim";

echo "Script has been done. [CNN-2]"
