'''
Author WHU ZFJ 2021
implementation of the TF-IDF baseline
'''
import json
from tqdm import tqdm
from elasticsearch import Elasticsearch, helpers


def for_line_in(file_path, length, bulk_size, func, *args, **kwargs):
    '''
    bulk processing lines with given function
    '''
    with open(file_path, mode='r', encoding='utf-8') as f:
        t = tqdm(total=length)
        line_no = 1
        lines = []
        for line in f:
            if len(lines) == bulk_size:
                func(lines, line_no, *args, **kwargs)
                t.update(bulk_size)
                lines = []
                line_no += bulk_size
            lines.append(line)
        if len(lines) != 0:
            func(lines, line_no, *args, **kwargs)
            t.update(len(lines))
        t.close()


def get_connection():
    es = Elasticsearch(f"http://{HOST}:9200")
    print("connected to elasticsearch")
    return es


def initialize(es):
    '''create the data index'''
    print(f"creating index: {INDEX_NAME}")
    result = es.indices.create(index=INDEX_NAME, ignore=400)
    print(result)


def delete_index(es):
    result = es.indices.delete(index=INDEX_NAME, ignore=[400, 404])
    print(result)


def query_index(es, field, support_str):
    '''
    return the first three matches
    '''
    data = {'query': {'match': {field: {'query': support_str,
                                        "analyzer": "standard"}}}}
    result = es.search(index=INDEX_NAME, doc_type=DOC_TYPE, body=data, size=3)
    top_three = result['hits']['hits']
    top_three_simple = \
        [{
            'id': hit['_id'],
            'score': hit['_score'],
            'body': hit['_source']['body'],
            'title': hit['_source']['title']
        } for hit in top_three]
    return top_three_simple


def convention_tokenize(text):
    special_tokens_id = list(range(33, 48))
    special_tokens_id += list(range(58, 65))
    special_tokens_id += list(range(91, 97))
    special_tokens_id += list(range(123, 127))
    special_tokens = [chr(i) for i in special_tokens_id]
    for st in special_tokens:
        text = f' {st} '.join(text.split(st)).strip()
    tokens = text.split()
    return tokens


def upload_json(lines, _, es):
    '''upload json document (used to train ccbert)'''
    bulk_actions = []
    for line in lines:
        line_json = json.loads(line.strip())
        title = ' '.join(line_json['target_tokens'])
        body = ' '.join(line_json['source_tokens'])
        action = {
            '_index': INDEX_NAME,
            '_type': DOC_TYPE,
            '_source': {
                'title': title,
                'body': body
            }
        }
        bulk_actions.append(action)
    helpers.bulk(es, bulk_actions, ignore=409)


def match_similar(es, source_path, target_path, ref_path, length):
    '''
    search for the most similar question
    write the searched question title to file
    '''
    t = tqdm(total=length)
    with open(source_path, mode='r', encoding='utf-8') as r_f,\
         open(target_path, mode='w', encoding='utf-8') as w_f,\
         open(ref_path, mode='w', encoding='utf-8') as e_f:
        line_no = 0
        for line in r_f:
            line_no += 1
            line = line.strip()
            line_json = json.loads(line)
            text_body = ' '.join(line_json['source_tokens'])
            title = ' '.join(line_json['target_tokens'])
            queryed = query_index(es, 'body', text_body)
            title_first = queryed[0]['title']
            w_f.write(f'{title_first}\n')
            e_f.write(f'{title}\n')
            t.update(1)
    t.close()


if __name__ == '__main__':
    INDEX_NAME = 'php_train'
    DOC_TYPE = 'php'
    FAILED_RECORD = 'failed.txt'  # log of the lines failed to upload
    BULK_SIZE = 100  # bulk_size
    HOST = '10.254.19.8'  # elastic database ip address
    es = get_connection()
    # delete_index(es)
    initialize(es)
    TOTAL_SIZE = 26535  # total number of lines to upload
    for_line_in("data/ccbert/php.both.train.jsonl", TOTAL_SIZE, BULK_SIZE, upload_json, es)
    match_similar(es, 'data/ccbert/php.both.valid.jsonl',
                  'data/elastic/pred/php.valid.match.all.txt',
                  'data/elastic/pred/php.valid.ref.txt', 1000)
