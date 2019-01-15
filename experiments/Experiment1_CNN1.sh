#!/usr/bin/env bash

# CNN experiment 1 [Learning rate]
python ../app.py --cell_type="CNN" --n_epoch=400 --learning_rate=1e-4 --dropout_keep_prob=0.7 --exp_name="exp01_1e-4";
python ../app.py --cell_type="CNN" --n_epoch=400 --learning_rate=1e-5 --dropout_keep_prob=0.7 --exp_name="exp01_1e-5";
python ../app.py --cell_type="CNN" --n_epoch=400 --learning_rate=1e-6 --dropout_keep_prob=0.7 --exp_name="exp01_1e-6";

# CNN experiment 2 [The number of conv-pool set]
python ../app.py --cell_type="CNN" --n_epoch=400 --learning_rate=1e-4 --filter_sizes=3,4 --dropout_keep_prob=0.7 --exp_name="exp02_2set";
python ../app.py --cell_type="CNN" --n_epoch=400 --learning_rate=1e-4 --filter_sizes=3,4,5 --dropout_keep_prob=0.7 --exp_name="exp02_3set";
python ../app.py --cell_type="CNN" --n_epoch=400 --learning_rate=1e-4 --filter_sizes=3,4,5,6 --dropout_keep_prob=0.7 --exp_name="exp02_4set";

echo "Script has been done. [CNN-1]"
