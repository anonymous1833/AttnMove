# -*- coding: utf-8 -*-
"""
configurate the hyperparameters, based on command line arguments.
"""
import argparse
import copy
import os

#from texar.data import SpecialTokens ##/texar/data/vocabulary.py
from dataset_utils import SpecialTokens

class Hyperparams:
    """
        config dictionrary, initialized as an empty object.
        The specific values are passed on with the ArgumentParser
    """
    def __init__(self):
        self.help = "the hyperparams dictionary to use"


def load_hyperparams():
    """
        main function to define hyperparams
    """
    # pylint: disable=too-many-statements
    args = Hyperparams()
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--mask_rate', type=float, default=0.25)
    argparser.add_argument('--blank_num', type=int, default=1)
    argparser.add_argument('--batch_size', type=int, default=3)
    argparser.add_argument('--test_batch_size', type=int, default=2)
    argparser.add_argument('--one_seq_length', type=int, default=48)
    argparser.add_argument('--hidden_dim', type=int, default=512)
    argparser.add_argument('--running_mode', type=str,
                           default='train_and_evaluate',
                           help='can also be test mode')
    argparser.add_argument('--history_util', type=str,
                           default='average_pooling',
                           help='can also be conv')
    argparser.add_argument('--gpu', type=str, default='7')
    argparser.add_argument('--process_name', type=str, default='TrajFill_seq48_3day_conv@qiyunhan')
    argparser.add_argument('--attn_way', type=str, default='cos_distance', help = 'can also be attn/RNN')
    argparser.add_argument('--history_num', type=int, default=3)
    argparser.add_argument('--reg_lambda', type=float, default=0.001)
    argparser.add_argument('--reg_lambda_l1', type=float, default=0.001)
    argparser.add_argument('--reg', type=int, default=1)
    argparser.add_argument('--reg_kind', type=str, default="l2", help = "l1 or l1_l2")
    argparser.add_argument('--rmse', type=int, default=1)
    argparser.add_argument('--rmse_factor', type=float, default=0.005)
    argparser.add_argument('--if_kernel', type=int, default=0)
    argparser.add_argument('--enhanced', type=int, default=1)
    argparser.add_argument('--if_mask', type=int, default=1)
    argparser.add_argument('--deepmove', type=int, default=1)
    argparser.add_argument('--if_bi_lstm', type=int, default=0)
    argparser.add_argument('--if_self_position', type=int, default=0)
    argparser.add_argument('--if_linear', type=int, default=0)
    argparser.add_argument('--if_history', type=int, default=0)
    argparser.add_argument('--structure', type=str, default="equal_connect", help = "structure of encoder and decoder")
    argparser.add_argument('--max_training_steps', type=int, default=2500000)
    argparser.add_argument('--warmup_steps', type=int, default=10000)
    argparser.add_argument('--max_train_epoch', type=int, default=500)
    argparser.add_argument('--bleu_interval', type=int, default=5)
    argparser.add_argument('--decay_interval', type=float, default=20)
    argparser.add_argument('--log_disk_dir', type=str, default='/data2/qiyunhan/textfill/')
    #argparser.add_argument('--log_disk_dir', type=str, default='analyse_data_save/')
    argparser.add_argument('--filename_prefix', type=str, default='pos.')
    argparser.add_argument('--data_dir', type=str, default='/data1/xiatong/trajen/pos_region_3dayhistory_small/')
    #argparser.add_argument('--data_dir', type=str, default='/data/')
    argparser.add_argument('--extra_data_dir', type=str, default='/data1/xiatong/trajen/region_extra/') # add a path for region distance
    argparser.add_argument('--save_eval_output', default=1,
                           help='save the eval output to file')
    argparser.add_argument('--lr_constant', type=float, default=0.3)
    argparser.add_argument('--lr', type=float, default=0.001)
    argparser.add_argument('--lr_decay_rate', type=float, default=0.5)
    argparser.add_argument('--lr_factor', type=float, default=0.1)
    argparser.add_argument('--learning_rate_strategy', type=str, default='dynamic')  # 'static'
    argparser.add_argument('--zero_pad', type=int, default=0)
    argparser.add_argument('--bos_pad', type=int, default=0,
                           help='use all-zero embedding for bos')
    argparser.add_argument('--random_seed', type=int, default=1234)
    argparser.add_argument('--beam_width', type=int, default=1)
    argparser.add_argument('--affine_bias', type=int, default=0)
    argparser.add_argument('--analyse_data_save_dir', type=str, default= "/data2/qiyunhan/textfill/model/seq48_region_3H/")
    #argparser.add_argument('--analyse_data_save_dir', type=str, default= "analyse_data_save/")
    argparser.add_argument('--nhead', type=int, default= 2)
    argparser.add_argument('--nlayer', type=int, default= 5)
    argparser.add_argument('--drop', type=float, default= 0.5)
    argparser.add_argument('--fb_drop', type=float, default= 0.5)
    argparser.add_argument('--train_user_file', type=str, default= "/data2/qiyunhan/textfill/yelp_data/pos/shuffled_train_uid.npy")
    #argparser.add_argument('--train_user_file', type=str, default= "../数据预处理/seq48_test/shuffled_train_uid.npy")
    argparser.add_argument('--test_user_file', type=str, default= "/data2/qiyunhan/textfill/yelp_data/pos/shuffled_test_uid.npy")
    #argparser.add_argument('--test_user_file', type=str, default= "../数据预处理/seq48_test/shuffled_test_uid.npy")
    argparser.parse_args(namespace=args)

    args.present_rate = 1 - args.mask_rate
    if args.enhanced == 1:
        args.max_seq_length = args.one_seq_length * (args.history_num + 3)
    else:
        args.max_seq_length = args.one_seq_length * (args.history_num + 1)
    print("max_seq_length = ", args.max_seq_length)
    args.max_decode_len = 1
    args.data_dir = os.path.abspath(args.data_dir)
    args.filename_suffix = '.txt'
    args.train_file = os.path.join(args.data_dir,
        '{}train{}'.format(args.filename_prefix, args.filename_suffix))
    args.valid_file = os.path.join(args.data_dir,
        '{}validate{}'.format(args.filename_prefix, args.filename_suffix))
    args.test_file = os.path.join(args.data_dir,
        '{}test{}'.format(args.filename_prefix, args.filename_suffix))

    args.vocab_file = os.path.join(args.data_dir, 'pos.vocab.txt')
    log_params_dir = 'log_dir/{}bsize{}.epoch{}.seqlen{}.{}_lr.present{}.partition{}.hidden{}.self_attn/'.format(
        args.filename_prefix, args.batch_size, args.max_train_epoch, args.max_seq_length,
        args.learning_rate_strategy, args.present_rate, args.blank_num, args.hidden_dim)
    args.log_dir = os.path.join(args.log_disk_dir, log_params_dir)
    print('train_file:{}'.format(args.train_file))
    print('valid_file:{}'.format(args.valid_file))
    print('test_file:{}'.format(args.test_file))
    print('vocab_file:{}'.format(args.vocab_file))
    train_dataset_hparams = {
        "num_epochs": 1,
        "seed": args.random_seed,
        "shuffle": False,
        "dataset": {
            "files": args.train_file,
            "vocab_file": args.vocab_file,
            "max_seq_length": args.max_seq_length,
            "bos_token": SpecialTokens.BOS,
            "eos_token": SpecialTokens.EOS,
            "length_filter_mode": "truncate",
        },
        'batch_size': args.batch_size,
        'allow_smaller_final_batch': False,
    }
    eval_dataset_hparams = {
        "num_epochs": 1,
        'seed': args.random_seed,
        'shuffle': False,
        'dataset' : {
            'files': args.valid_file,
            'vocab_file': args.vocab_file,
            "max_seq_length": args.max_seq_length,
            "bos_token": SpecialTokens.BOS,
            "eos_token": SpecialTokens.EOS,
            "length_filter_mode": "truncate",
        },
        'batch_size': args.batch_size,
        'allow_smaller_final_batch': False,
    }
    test_dataset_hparams = {
        "num_epochs": 1,
        "seed": args.random_seed,
        "shuffle": False,
        "dataset": {
            "files": args.test_file,
            "vocab_file": args.vocab_file,
            "max_seq_length": args.max_seq_length,
            "bos_token": SpecialTokens.BOS,
            "eos_token": SpecialTokens.EOS,
            "length_filter_mode": "truncate",
        },
        'batch_size': args.test_batch_size,
        'allow_smaller_final_batch': False,
    }
    args.word_embedding_hparams = {
        'name': 'lookup_table',
        'dim': args.hidden_dim,
        'initializer': {
            'type': 'random_normal_initializer',
            'kwargs': {
                'mean': 0.0,
                'stddev': args.hidden_dim**-0.5,
            },
        }
    }
    encoder_hparams = {
        'multiply_embedding_mode': "sqrt_depth",
        'embedding_dropout': 0.1,
        'position_embedder': {
            'name': 'sinusoids',
            'hparams': None,
        },
        'attention_dropout': args.drop,
        'residual_dropout': args.drop,
        'fb_dropout':args.fb_drop,
        'if_kernel':args.if_kernel,
        'if_mask':args.if_mask,
        'if_bi_lstm':args.if_bi_lstm,
        'blank_num':args.blank_num,
        'if_history':args.if_history,
        'batch_size':args.batch_size,
        'one_seq_length':args.one_seq_length,
        'deepmove':args.deepmove,
        'if_self_position':args.if_self_position,
        'history_num':args.history_num,
        'attn_way':args.attn_way,
        'structure':args.structure,
        'sinusoid': True,
        'num_blocks': args.nlayer,
        'num_heads': args.nhead,
        'num_units': args.hidden_dim,
        'zero_pad': args.zero_pad,
        'bos_pad': args.bos_pad,
        "hidden_dim":args.hidden_dim,
        "history_util":args.history_util,
        'initializer': {
            'type': 'variance_scaling_initializer',
            'kwargs': {
                'scale': 1.0,
                'mode': 'fan_avg',
                'distribution':'uniform',
            },
        },
        'poswise_feedforward': {
            'name': 'ffn',
            'layers': [
                {
                    'type': 'Dense',
                    'kwargs': {
                        'name': 'conv1',
                        'units': args.hidden_dim*4,
                        'activation': 'relu',
                        'use_bias': True,
                    }
                },
                {
                    'type': 'Dropout',
                    'kwargs': {
                        'rate': 0.1,
                    }
                },
                {
                    'type':'Dense',
                    'kwargs': {
                        'name': 'conv2',
                        'units': args.hidden_dim,
                        'use_bias': True,
                        }
                }
            ],
        },
    }
    decoder_hparams = copy.deepcopy(encoder_hparams)
    decoder_hparams['share_embed_and_transform'] = True
    decoder_hparams['transform_with_bias'] = args.affine_bias
    decoder_hparams['maximum_decode_length'] = args.max_decode_len
    decoder_hparams['beam_width'] = args.beam_width
    decoder_hparams['sampling_method'] = 'argmax'
    loss_hparams = {
        'label_confidence': 0.9,
    }

    opt_hparams = {
        'learning_rate_schedule': args.learning_rate_strategy,
        'lr_constant': args.lr_constant,
        'warmup_steps': args.warmup_steps,
        'max_training_steps': args.max_training_steps,
        'Adam_beta1': 0.9,
        'Adam_beta2': 0.997,
        'Adam_epsilon': 1e-9,
    }
    opt_vars = {
        'learning_rate': args.lr,  # 0.016
        'best_train_loss': 1e100,
        'best_eval_loss': 1e100,
        'best_eval_bleu': 0,
        'steps_not_improved': 0,
        'epochs_not_improved': 0,
        'decay_interval': args.decay_interval,
        'lr_decay_rate': args.lr_decay_rate,
        'decay_time': 0
    }
    print('logdir:{}'.format(args.log_dir))
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.log_dir + 'img/'):
        os.makedirs(args.log_dir + 'img/')
    return {
        'train_dataset_hparams': train_dataset_hparams,
        'eval_dataset_hparams': eval_dataset_hparams,
        'test_dataset_hparams': test_dataset_hparams,
        'encoder_hparams': encoder_hparams,
        'decoder_hparams': decoder_hparams,
        'loss_hparams': loss_hparams,
        'opt_hparams': opt_hparams,
        'opt_vars': opt_vars,
        'args': args,
        }
