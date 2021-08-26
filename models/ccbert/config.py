'''
Author WHU ZFJ 2021
all the configs for CCBERT
'''
import argparse
import logging
import torch
import os


parser = argparse.ArgumentParser()
version_id = 'java_both'
# file paths
DATA_FILE = 'data/java.both.{}.jsonl'
parser.add_argument("--train_file_path", default=DATA_FILE.format('train'), type=str)
parser.add_argument("--val_file_path", default=DATA_FILE.format('valid'), type=str)
parser.add_argument("--test_file_path", default=DATA_FILE.format('test'), type=str)
parser.add_argument("--gen_gold_path", default=f'{version_id}.golden.txt', type=str)
parser.add_argument("--gen_pred_path", default=f'{version_id}.pred.txt', type=str)
# dataset related
parser.add_argument("--max_src_len", default=512, type=int)
parser.add_argument("--max_tgt_len", default=64, type=int)
parser.add_argument("--train_batch_size", default=8, type=int)
parser.add_argument("--test_batch_size", default=64, type=int)
# model related
parser.add_argument("--learning_rate", default=5e-5, type=float)
parser.add_argument("--dec_layers", default=6, type=int)
parser.add_argument("--beam_size", default=10, type=int)
parser.add_argument("--do_lower_case", default=True, type=bool)
parser.add_argument("--weight_decay", default=0.0, type=float)
parser.add_argument("--model_name", default="microsoft/codebert-base", type=str)
# logs and some other frequently changing args
parser.add_argument("--best_rouge_model_save_path", default=f'{version_id}.brouge.pt', type=str)
parser.add_argument("--log_file", default=f'logs/{version_id}.log', type=str)
parser.add_argument("--load_model_from_epoch", default=0, type=int)
parser.add_argument("--last_best_rouge", default=0.2, type=float)
parser.add_argument("--epoch_model_save_path", default=f'{version_id}.pt', type=str)
parser.add_argument("--load_model_path", default=f'{version_id}.brouge.pt', type=str)
# training related
parser.add_argument("--gpu", default="0,1", type=str)
parser.add_argument("--random_seed", default=999, type=int)
parser.add_argument("--epoch", default=15, type=int)
parser.add_argument("--gradient_accumulation_steps", default=4, type=int)
parser.add_argument("--debug", default=0, type=int)
parser.add_argument("--generate", default=0, type=int)
parser.add_argument("--train", default=1, type=int)
parser.add_argument("--log_interval", default=100, type=int)
parser.add_argument("--score_interval", default=200, type=int)
parser.add_argument("--eval_interval", default=1750, type=int)
args = parser.parse_args()

'''make directories'''
for directory in ['./logs', './state_dicts']:
    if not os.path.exists(directory):
        os.makedirs(directory)

'''logging'''
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s \t %(message)s')
handler = logging.FileHandler(args.log_file, 'a', 'utf-8')
handler.setFormatter(formatter)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
if not args.debug:
    logger.addHandler(handler)
logger.addHandler(console_handler)

'''print all args'''
logger.info(args)

'''for multi gpu'''
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
gpu_count = len(args.gpu.split(','))
logger.info(f'GPU count {gpu_count}, no. {args.gpu}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')