# STtransformer
This is the code for submission #7726 in AAAI2020 


## Datasets

The geolife data to evaluate our model, which contains 50 users and ready for directly used. 

## Requirements

- Python 3.0
- TF-1.12 
- Keras for baseline

## Project Structure

- /data # preprocessed  
- /codes 
  - main.py 
  - model.py # define models
  - train.py # define tools for train the model
- /baseline #codes for baselines 
  - main.py 
  - model.py # define models
  - train.py # define tools for train the model

## Usage

```
python main.py --users_end 1000 --model_mode AppPreLocPreUserIdenGtr --lr_step 1 --process_name user_iden_alpha_beta --hidden_size 512 --app_encoder_size 512 --loss_beta 0.2 --loss_alpha 0.2  
```

The codes contain four network model (DeepApp, DeepApp(App), DeepApp(App+Loc), DeepApp(App+User), RNN) and baseline model (MRU, MFU, HA, Bayes). The parameter settings for these model can refer to run.sh file. 

