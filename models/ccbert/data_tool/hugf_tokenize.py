'''
Author WHU ZFJ 2021
data pre- and post- processing
'''
import torch
import linecache
import json
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.nn.utils.rnn import pad_sequence
from transformers import RobertaTokenizer
from config import args
from data_tool.scorer import convention_tokenize


tokenizer = RobertaTokenizer.from_pretrained(args.model_name, do_lower_case=args.do_lower_case)

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self, example_id, source_ids,
                 target_ids):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids


def read_features(filename, args):
    """
    Convert source data to features
    """
    features = []
    lines = linecache.getlines(filename)
    for idx, line in enumerate(lines):
        line = line.strip()
        js = json.loads(line)
        code = ' '.join(js['source_tokens']).lower()
        nl = ' '.join(js['target_tokens']).lower()
        example_id = idx
        source_tokens = tokenizer.tokenize(code)[:args.max_src_len-2]
        source_tokens = [tokenizer.cls_token] + source_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        
        target_tokens = tokenizer.tokenize(nl)[:args.max_tgt_len-2]
        target_tokens = [tokenizer.bos_token] + target_tokens + [tokenizer.eos_token]
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        features.append(InputFeatures(
            example_id, source_ids, target_ids
        ))
    return features

def gen_batch(data_batch):
    '''
    Padding according to the longest sequence in the batch
    The padding is not long enough, and the truncation is too long
    Return the id sequence after padding 
    '''
    def _trunc_and_pad(seq_ids, max_len):
        # seq_ids SOS, EOS, UNK already included 
        eos_id = seq_ids[-1]
        seq_ids = seq_ids[:-1] # Delete EOS first 
        seq_ids = list(seq_ids[:max_len-1]) # Truncate data that is too long 
        seq_ids.append(eos_id) # Add EOS to the truncated data 
        seq_ids = torch.LongTensor(seq_ids)
        return seq_ids
        
    src_ids_for_padding = []
    tgt_ids_for_padding = []
    for raw_src_ids_batch, raw_tgt_ids_batch in data_batch:
        # raw_src_ids_batch [seq_len]
        src_ids_batch = _trunc_and_pad(raw_src_ids_batch, args.max_src_len)
        target_ids_batch = _trunc_and_pad(raw_tgt_ids_batch, args.max_tgt_len)
        src_ids_for_padding.append(src_ids_batch)
        tgt_ids_for_padding.append(target_ids_batch)
    padded_src_ids = pad_sequence(src_ids_for_padding, padding_value=tokenizer.pad_token_id) # [seq_len, batch_size]
    padded_tgt_ids = pad_sequence(tgt_ids_for_padding, padding_value=tokenizer.pad_token_id)
    # result = [i for i in zip(padded_src_ids, padded_tgt_ids)]
    src_padding_mask = torch.ones(padded_src_ids.shape) # The pad is 0 
    src_padding_mask[padded_src_ids==tokenizer.pad_token_id] = 0
    tgt_padding_mask = torch.ones(padded_tgt_ids.shape) # The pad is 0 
    tgt_padding_mask[padded_tgt_ids==tokenizer.pad_token_id] = 0
    return padded_src_ids, src_padding_mask, padded_tgt_ids, tgt_padding_mask

def get_dataloaders(args):
    '''
    The ultimate encapsulation of the data layer
    read the divided data set, and convert it to the dataloader of torch 
    '''
    train_file_path = args.train_file_path
    val_file_path = args.val_file_path
    test_file_path = args.test_file_path
    train_batch_size = args.train_batch_size
    test_batch_size = args.test_batch_size
    
    datasets = [] # train, val, test
    for data_file in [train_file_path, val_file_path, test_file_path]:
        features = read_features(data_file, args)
        source_ids = [torch.tensor(f.source_ids) for f in features]
        target_ids = [torch.tensor(f.target_ids) for f in features]
        dataset = [i for i in zip(source_ids, target_ids)]
        datasets.append(dataset)
    train_sampler = RandomSampler(datasets[0])
    train_dataloader = DataLoader(datasets[0], sampler=train_sampler, batch_size=train_batch_size, collate_fn=gen_batch)
    val_sampler = SequentialSampler(datasets[1])
    val_dataloader = DataLoader(datasets[1], sampler=val_sampler, batch_size=test_batch_size, collate_fn=gen_batch)
    test_sampler = SequentialSampler(datasets[2])
    test_dataloader = DataLoader(datasets[2], sampler=test_sampler, batch_size=test_batch_size, collate_fn=gen_batch)
    return train_dataloader, val_dataloader, test_dataloader


def decode_batch_ids(batch_seq_ids):
    '''
    batch_seq_ids [batch_size, seq_len]
    '''
    pred_tokens = []
    for seq in batch_seq_ids:
        token_ids = []
        for tok_id in seq:
            if tok_id == tokenizer.eos_token_id:
                break
            if tok_id == tokenizer.bos_token_id:
                break
            token_ids.append(tok_id.item())
        decode_tokens = tokenizer.decode(token_ids)
        decode_tokens = convention_tokenize(decode_tokens)
        pred_tokens.append(decode_tokens)
    return pred_tokens
