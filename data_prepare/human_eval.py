'''
Author WHU ZFJ 2022
Semi-automated pipeline for human evaluation
'''
import ipdb
import linecache
import numpy as np
from collections import Counter

from data_tools.utils import save_cache, load_cache

php_target = 'data/onmt/php.both.valid.tgt.txt'
js_target = 'data/onmt/javascript.both.valid.tgt.txt'

php_tf_idf = 'data/elastic/pred/php.valid.match.all.txt'
js_tf_idf = 'data/elastic/pred/javascript.valid.match.all.txt'

php_lscc = 'data/onmt/pred/lstm_cc_four_php.valid.pred.txt'
js_lscc = 'data/onmt/pred/lstm_cc_four_javascript.valid.pred.txt'

php_ccbert = 'data/bart/pred/24400-four_lang_php.valid.pred.txt'
js_ccbert = 'data/bart/pred/24400-four_lang_javascript.valid.pred.txt'

def pick_questions():
    js_line_nos = np.arange(2000)
    js_picked = np.random.choice(js_line_nos, 320, replace=False)
    php_line_nos = np.arange(1000)
    php_picked = np.random.choice(php_line_nos, 180, replace=False)
    
    js_tgt_lines = linecache.getlines(js_target)
    js_tf_pred = linecache.getlines(js_tf_idf)
    js_lscc_pred = linecache.getlines(js_lscc)
    js_ccbert_pred = linecache.getlines(js_ccbert)
    js_picked_tgt = dict()
    js_picked_tf = dict()
    js_picked_lstm = dict()
    js_picked_bert = dict()

    for i in js_picked:
        try:
            js_picked_tgt[i] = js_tgt_lines[i]
            js_picked_tf[i] = js_tf_pred[i]
            js_picked_lstm[i] = js_lscc_pred[i]
            js_picked_bert[i] = js_ccbert_pred[i]
        except Exception:
            ipdb.set_trace()
    
    php_tgt_lines = linecache.getlines(php_target)
    php_tf_pred = linecache.getlines(php_tf_idf)
    php_lscc_pred = linecache.getlines(php_lscc)
    php_ccbert_pred = linecache.getlines(php_ccbert)
    php_picked_tgt = dict()
    php_picked_tf = dict()
    php_picked_lstm = dict()
    php_picked_bert = dict()

    for i in php_picked:
        php_picked_tgt[i] = php_tgt_lines[i]
        php_picked_tf[i] = php_tf_pred[i]
        php_picked_lstm[i] = php_lscc_pred[i]
        php_picked_bert[i] = php_ccbert_pred[i]
    
    return {
        'js': {
            'idx': js_picked,
            'target': js_picked_tgt,
            'tf-idf': js_picked_tf,
            'lstmcc': js_picked_lstm,
            'ccbert': js_picked_bert
        },
        'php': {
            'idx': php_picked,
            'target': php_picked_tgt,
            'tf-idf': php_picked_tf,
            'lstmcc': php_picked_lstm,
            'ccbert': php_picked_bert
        }
    }

def rate_questions(picked, rated):
    rated_cache_path = 'data/human/rated.pkl'
    rated_js_idx = rated['js']['idx']
    for i in picked['js']['idx']:
        if i in rated_js_idx:
            continue
        print(f'\n-----------id {i}---------------')
        print(f'target|| {picked["js"]["target"][i]}')
        print(f'tf-idf|| {picked["js"]["tf-idf"][i]}')
        print(f'lstm-c|| {picked["js"]["lstmcc"][i]}')
        print(f'ccbert|| {picked["js"]["ccbert"][i]}')
        print('--------------rate---------------')
        tf = input("tf-idf:").split(' ')
        lstm = input("lstm-c:").split(' ')
        bert = input("ccbert:").split(' ')
        rated['js']['tf_scores'][i] = [int(j) for j in tf],
        rated['js']['ls_scores'][i] = [int(j) for j in lstm],
        rated['js']['bert_scores'][i] = [int(j) for j in bert]
        rated['js']['idx'].add(i)
        save_cache(rated, rated_cache_path)
        print(f"{len(rated['js']['idx'])+len(rated['php']['idx'])} rated!")

    rated_php_idx = rated['php']['idx']
    for i in picked['php']['idx']:
        if i in rated_php_idx:
            continue
        print(f'\n-----------id {i}---------------')
        print(f'target|| {picked["php"]["target"][i]}')
        print(f'tf-idf|| {picked["php"]["tf-idf"][i]}')
        print(f'lstm-c|| {picked["php"]["lstmcc"][i]}')
        print(f'ccbert|| {picked["php"]["ccbert"][i]}')
        print('--------------rate---------------')
        tf = input("tf-idf:").split(' ')
        lstm = input("lstm-c:").split(' ')
        bert = input("ccbert:").split(' ')
        rated['php']['tf_scores'][i] = [int(j) for j in tf],
        rated['php']['ls_scores'][i] = [int(j) for j in lstm],
        rated['php']['bert_scores'][i] = [int(j) for j in bert]
        rated['php']['idx'].add(i)
        save_cache(rated, rated_cache_path)
        print(f"{len(rated['js']['idx'])+len(rated['php']['idx'])} rated!")

def show_rated(rated):
    def _show_one_model(scores):
        total = len(scores)
        try:
            readability = [scores[key][0][0] for key in scores.keys()]
            correlation = [scores[key][0][1] for key in scores.keys()]
        except Exception:
            readability = [scores[key][0] for key in scores.keys()]
            correlation = [scores[key][1] for key in scores.keys()]
        readability_counter = Counter(readability)
        readability_dict = dict()
        for key in readability_counter.keys():
            readability_dict[key] = readability_counter[key] / total
        correlation_counter = Counter(correlation)
        correlation_dict = dict()
        for key in correlation_counter.keys():
            correlation_dict[key] = correlation_counter[key] / total
        print(f'readability: {readability_dict} avg: {sum(readability)/total}')
        print(f'correlation: {correlation_dict} avg: {sum(correlation)/total}')
    
    print(f'-----TF-IDF-----')
    _show_one_model(rated['js']['tf_scores'])
    print(f'-----BiLSTM-CC-----')
    _show_one_model(rated['js']['ls_scores'])
    print(f'-----CCBERT-----')
    _show_one_model(rated['js']['bert_scores'])

            

if __name__ == '__main__':
    picked_cache_path = 'data/human/picked.pkl'
    picked = load_cache(picked_cache_path)
    if not picked:
        picked = pick_questions()
        save_cache(picked, picked_cache_path)

    rated_cache_path = 'data/human/rated.pkl'
    rated = load_cache(rated_cache_path)
    if not rated:
        rated = {
            'js': {
                'idx': set(),
                'tf_scores': dict(),
                'ls_scores': dict(),
                'bert_scores': dict()
            },
            'php': {
                'idx': set(),
                'tf_scores': dict(),
                'ls_scores': dict(),
                'bert_scores': dict()
            }
        }
    # rate_questions(picked, rated)
    show_rated(rated)

    