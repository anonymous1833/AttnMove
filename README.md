# AttnMove
This is the code for submission #9808 in AAAI2020 


## Datasets

The geolife data to evaluate our model, which contains 40 users and ready for directly used. 

## Requirements

- Python 3.6
- TF-1.18

## Project Structure

- /data # preprocessed
  - pos.train.txt
  - pos.validate.txt
  - pos.test.txt
  - pos,vocab.txt
- /model # AttnMove
  - main.py
  - dataset_utils.py
  - Embedder.py
  - model_utils.py
  - self_attn_hyperaram_region.py
- /baseline #codes for baselines 
  - baseline.py
  - dataset_utils.py
  - Embedder.py
  - model_utils.py
  - self_attn_hyperaram_region.py

## Usage for AttnMove

```
python3 main.py --one_seq_length 48 --blank_num 10 --batch_size 50 --learning_rate_strategy static --max_train_epoch 1000 --hidden_dim 64 --bleu_interval 5 --nhead 1 --nlayer 2 --test_batch_size 50 --reg_lambda 1e-2 --gpu 4 --if_mask 1 --data_dir ../data/ --if_history 1 --lr 0.001 --drop 0.3 --fb_drop 0.3 --lr_decay_rate 0.1
```

The codes contain four network model (AttnMove, AttnMove-H) and baseline model (LSTM, Bi-LSTM, DeepMove). 
