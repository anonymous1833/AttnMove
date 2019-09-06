This is the code for our proposed model(AttnMove).

## Usage for AttnMove
```
python3 main.py --blank_num 10 --if_history 1 --hidden_dim 64 --bleu_interval 5 --nhead 1 --nlayer 2 --reg_lambda 1e-2 --gpu 4 --if_mask 1 --drop 0.3 --fb_drop 0.3
```

## Usage for AttnMove-H
```
python3 main.py --blank_num 10 --if_history 0 --hidden_dim 64 --bleu_interval 5 --nhead 1 --nlayer 2 --reg_lambda 1e-2 --gpu 4 --if_mask 1 --drop 0.3 --fb_drop 0.3
```

