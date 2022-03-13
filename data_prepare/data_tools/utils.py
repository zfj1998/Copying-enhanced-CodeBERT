'''
Author WHU ZFJ 2021
tools involving file system and tokenizing
'''
from tqdm import tqdm
import pickle
import os


POSTS_ALL_XML_LINE_COUNT = 51296934
GOOD_QUESTIONS_LINE_COUNT = 3213260
GOOD_QUESTIONS_2019_2020 = 241352
BATCH_SIZE = 1000


def line_counter(source_path):
    '''
    a simple tool to count lines of a file
    '''
    if 'Posts.xml' in source_path:
        return POSTS_ALL_XML_LINE_COUNT
    if 'Good_questions.jsonl' in source_path:
        return GOOD_QUESTIONS_LINE_COUNT
    if 'Good_questions_from_2019_to_2020.jsonl' in source_path:
        return GOOD_QUESTIONS_2019_2020
    count = 0
    with open(source_path, 'r', encoding='utf-8') as f:
        for _ in tqdm(f):
            count += 1
    return count


def write_to_file(lines, file_path, mode='w'):
    with open(file_path, mode, encoding='utf-8') as f:
        f.writelines(lines)


def save_cache(obj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)


def load_cache(file_path):
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def convention_tokenize(text):
    '''
    split by all special tokens in ascii
    '''
    special_tokens_id = list(range(33, 48))
    special_tokens_id += list(range(58, 65))
    special_tokens_id += list(range(91, 97))
    special_tokens_id += list(range(123, 127))
    special_tokens = [chr(i) for i in special_tokens_id]
    for st in special_tokens:
        text = f' {st} '.join(text.split(st)).strip()
    tokens = text.split()
    return tokens
