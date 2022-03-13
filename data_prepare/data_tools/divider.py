'''
Author WHU ZFJ 2021
Split datasets into train, validate and test subsets
'''
import random
import numpy as np
import math
from tqdm import tqdm
from data_tools.utils import line_counter, write_to_file


random.seed(99)
np.random.seed(99)


def partition_on_time(source_path, test_samples=4000):
    '''
    split dataset based on datetime(already datetime-ordered in source_path file)
    paths: include source_file, tgt_train_path, tgt_valid_path, tgt_test_path
    '''
    train_path = source_path.replace('.jsonl', '.train.jsonl')
    test_path = source_path.replace('.jsonl', '.test.jsonl')
    valid_path = source_path.replace('.jsonl', '.valid.jsonl')

    total_lines_for_test = test_samples
    test_size = total_lines_for_test // 2
    # source file follows the order of datetime
    total_line = line_counter(source_path)
    for_test = np.arange(total_line-total_lines_for_test, total_line)
    test_ids = np.random.choice(for_test, test_size, replace=False)
    for_test = set(for_test)
    test_ids = set(test_ids)
    valid_ids = for_test - test_ids
    train_lines = []
    test_lines = []
    valid_lines = []
    with open(source_path, 'r', encoding='utf-8') as f:
        t = tqdm(total=total_line)
        for line_id, line in enumerate(f):
            t.update(1)
            if line_id in test_ids:
                test_lines.append(line)
            elif line_id in valid_ids:
                valid_lines.append(line)
            else:
                train_lines.append(line)
        t.close()
    random.shuffle(train_lines)
    write_to_file(train_lines, train_path)
    write_to_file(test_lines, test_path)
    write_to_file(valid_lines, valid_path)


def mix_datasets(source_paths, target_path, partition):
    '''
    mix train sets of different programming languages together and randomly shuffle.
    validation and testing sets are conconated sequently
    '''
    train_lines_total = []
    test_lines = []
    valid_lines = []
    for source_path in source_paths:
        train_lines = []
        train_path = source_path.replace('.jsonl', '.train.jsonl')
        test_path = source_path.replace('.jsonl', '.test.jsonl')
        valid_path = source_path.replace('.jsonl', '.valid.jsonl')
        with open(train_path, 'r', encoding='utf-8') as f:
            for line in f:
                train_lines.append(line)
        random.shuffle(train_lines)
        lines_to_take = math.ceil(len(train_lines) / partition)
        train_lines_total += train_lines[:lines_to_take]
        with open(test_path, 'r', encoding='utf-8') as f:
            for line in f:
                test_lines.append(line)
        with open(valid_path, 'r', encoding='utf-8') as f:
            for line in f:
                valid_lines.append(line)
    random.shuffle(train_lines_total)
    train_path = target_path.replace('.jsonl', '.train.jsonl')
    test_path = target_path.replace('.jsonl', '.test.jsonl')
    valid_path = target_path.replace('.jsonl', '.valid.jsonl')
    write_to_file(train_lines_total, train_path)
    write_to_file(test_lines, test_path)
    write_to_file(valid_lines, valid_path)
