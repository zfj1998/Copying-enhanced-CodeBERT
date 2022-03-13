'''
Author WHU ZFJ 2021
a simple way to implement the Oracle method
'''
import json
from tqdm import tqdm

def write_lines(file_path, content):
    with open(file_path, 'a', encoding='utf-8') as f:
        f.writelines(content)

def file_prediction(source_path, pred_path, total_line_count):
    '''
    given source_file (used for ccbert), generate candidate prediction
    '''
    result = []
    with open(source_path, 'r', encoding='utf-8') as f:
        t = tqdm(total=total_line_count)

        for line in f:
            t.update(1)
            json_line = json.loads(line.strip())
            content = json_line['source_tokens']
            content = set(content)
            reference = json_line['target_tokens']
            prediction = ''
            for token in reference:
                if token not in content:
                    continue
                prediction = f'{prediction} {token}'
            result.append(f'{prediction}\n')
        write_lines(pred_path, result)
        t.close()

if __name__ == '__main__':
    source_path = 'data/ccbert/php.both.valid.jsonl'
    line_count = 2000
    pred_path = 'models/oracle/php.valid.pred.txt'
    file_prediction(source_path, pred_path, line_count)
