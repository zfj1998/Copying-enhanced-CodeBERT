'''
Author WHU ZFJ 2021
Make datasets for different models
'''
import linecache
import json
from itertools import product

from data_tools.utils import convention_tokenize, write_to_file


SOURCE_PATH = 'data/json/w_wo_interrogative_{language}_questions.{partition}.jsonl'


def merge_segments(content, text_only=False, code_only=False):
    '''
    merge segments of code snippets and text descriptions
    '''
    integrated = ''
    for i in content:
        if code_only and i[0] == 'code':
            integrated += ' ' + i[1]
            continue
        if text_only and i[0] == 'text':
            integrated += ' ' + i[1]
            continue
        if (not code_only) and (not text_only):
            integrated += ' ' + i[1]
    return integrated


def build_for_onmt():
    '''
    onmt model needs lines of string for both input and output
    but the source input and target need be put in different files
    - all the lines are tokenized and turned into lower case
    '''
    onmt_file = 'data/onmt/{language}.w_wo.{type}.{partition}.{direction}.txt'
    segment = {
        'code_only': (False, True),
        'both': (False, False)
    }
    directions = {
        'src': '@Body',
        'tgt': '@Title'
    }
    configs = product(
        ['four_lang'],
        ['both'],
        ['train', 'valid', 'test'],
        ['src', 'tgt']
    )
    # configs = product(
    #     ['python', 'java'],
    #     ['code_only', 'both'],
    #     ['train', 'valid', 'test'],
    #     ['src', 'tgt']
    # )
    for (lang, type, part, direct) in configs:
        source_path = SOURCE_PATH.format(
            language=lang,
            partition=part
        )
        target_path = onmt_file.format(
            language=lang,
            type=type,
            partition=part,
            direction=direct
        )
        print(f'read {source_path}')
        raw_lines = linecache.getlines(source_path)
        key = directions[direct]

        def _handle(line):
            js_line = json.loads(line.strip())
            content = js_line[key]
            if direct == 'src':
                seg_cfg = segment[type]
                content = merge_segments(content, *seg_cfg)
            content = content.lower()
            content = ' '.join(convention_tokenize(content))
            return f'{content}\n'
        print(f'start writing {target_path}')
        extracted = [_handle(i) for i in raw_lines]
        write_to_file(extracted, target_path)


def build_for_ccbert():
    '''
    ccbert needs json lines of both source tokens and target tokens
    - all the lines are tokenized and turned into lower case
    '''
    onmt_file = 'data/ccbert/{language}.{type}.{partition}.jsonl'
    segment = {
        'code_only': (False, True),
        'both': (False, False)
    }
    # configs = product(
    #     ['python', 'java'],
    #     ['code_only', 'both'],
    #     ['train', 'valid', 'test']
    # )
    configs = product(
        ['four_lang'],
        ['code_only'],
        ['train', 'valid', 'test']
    )
    for (lang, type, part) in configs:
        source_path = SOURCE_PATH.format(
            language=lang,
            partition=part
        )
        target_path = onmt_file.format(
            language=lang,
            type=type,
            partition=part
        )
        print(f'read {source_path}')
        raw_lines = linecache.getlines(source_path)

        def _handle(line):
            js_line = json.loads(line.strip())
            body = js_line['@Body']
            title = js_line['@Title']
            seg_cfg = segment[type]
            body = merge_segments(body, *seg_cfg)
            title = convention_tokenize(title.lower())
            body = convention_tokenize(body.lower())
            js_result = {
                'source_tokens': body,
                'target_tokens': title
            }
            return f'{json.dumps(js_result)}\n'
        print(f'start writing {target_path}')
        extracted = [_handle(i) for i in raw_lines]
        write_to_file(extracted, target_path)


def build_for_bart():
    '''
    almost the same as ccbert
    '''
    onmt_file = 'data/bart/{language}.w_wo.{partition}.jsonl'
    configs = product(
        ['four_lang'],
        ['train', 'valid', 'test']
    )
    for (lang, part) in configs:
        source_path = SOURCE_PATH.format(
            language=lang,
            partition=part
        )
        target_path = onmt_file.format(
            language=lang,
            partition=part
        )
        print(f'read {source_path}')
        raw_lines = linecache.getlines(source_path)

        def _handle(line):
            js_line = json.loads(line.strip())
            body = js_line['@Body']
            title = js_line['@Title']
            body = merge_segments(body, False, False)
            title = convention_tokenize(title.lower())
            body = convention_tokenize(body.lower())
            js_result = {
                'code_tokens': body,
                'docstring_tokens': title
            }
            return f'{json.dumps(js_result)}\n'
        print(f'start writing {target_path}')
        extracted = [_handle(i) for i in raw_lines]
        write_to_file(extracted, target_path)


if __name__ == '__main__':
    build_for_onmt()
    # build_for_ccbert()
    build_for_bart()
