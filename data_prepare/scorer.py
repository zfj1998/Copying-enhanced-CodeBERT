'''
Author WHU ZFJ 2021
Score with BLEU, Rouge
Scored with lower case, tokenize in our own way
'''
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge


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


def get_word_maps_from_file(pred_file, gold_file, tokenize):
    '''
    The following line formats are acceptable:
        {idx}\t{content}
        {content}
    The word segmentation method can be customized, as long as
    the content is a string consisting of tokens separated by spaces
    return:
        hypotheses: dict{id: [str]}
        references: dict{id: [str]}
    '''
    prediction_map = {}
    gold_map = {}
    with open(pred_file, 'r', encoding='utf-8') as pf, \
         open(gold_file, 'r', encoding='utf-8') as gf:
        for row_no, row in enumerate(pf):
            cols = row.strip()
            rid, pred = row_no, cols
            if not pred:
                pred = 'nothinggenerated andscoreshouldbezero'
            if len(pred.split(' ')) == 1:
                pred = f'{pred} {pred}'
            prediction_map[rid] = [pred.strip().lower()]
            if tokenize:
                tokenized_pred = convention_tokenize(prediction_map[rid][0])
                prediction_map[rid] = [' '.join(tokenized_pred)]

        for row_no, row in enumerate(gf):
            cols = row.strip()
            rid, gold = row_no, cols
            gold_map[rid] = [(gold.strip().lower())]
            if tokenize:
                tokenized_gold = convention_tokenize(gold_map[rid][0])
                gold_map[rid] = [' '.join(tokenized_gold)]

        print(f'Total: {len(gold_map)}')
        return (prediction_map, gold_map)


def nltk_bleu(hypotheses, references):
    '''return float'''
    total_score = 0
    count = len(hypotheses)
    smoothing = SmoothingFunction().method2
    for key in list(hypotheses.keys()):
        hyp = hypotheses[key][0].split()
        ref = [r.split() for r in references[key]]
        score = sentence_bleu(ref, hyp, weights=(0.25, 0.25, 0.25, 0.25),
                              smoothing_function=smoothing)
        total_score += score
    avg_score = total_score / count
    return avg_score


def py_rouge(hypotheses, references):
    '''
    {'rouge-1': {'f':, 'p':, 'r':}, 'rouge-2', 'rouge-l'}
    '''
    rouge = Rouge()
    hyps = []
    refs = []
    for key in list(hypotheses.keys()):
        hyps.append(hypotheses[key][0])
        refs.append(references[key][0])
    scores = rouge.get_scores(hyps, refs, avg=True)
    return scores


def average_rouge(hypotheses, references):
    '''
    return the avg f score of rouge1, rouge2, rougeL
    '''
    scores = py_rouge(hypotheses, references)
    rouge1_f = scores['rouge-1']['f']
    rouge2_f = scores['rouge-2']['f']
    rougeL_f = scores['rouge-l']['f']
    avg_f = (rouge1_f + rouge2_f + rougeL_f)/3
    return avg_f, scores


def running_local(pred_path, gold_path):
    hypotheses, references = get_word_maps_from_file(pred_path,
                                                     gold_path, True)
    bleu = nltk_bleu(hypotheses, references)
    rouge_score = average_rouge(hypotheses, references)
    print(f'bleu:\n{bleu}')
    print(f'rouge:\n{rouge_score}')

def cal_four_lang():
    languages = ['java', 'javascript', 'python', 'php']
    pred_path = 'data/bart/pred/bart_w_wo_102000.valid.pred.txt'
    gold_path = 'data/onmt/four_lang.w_wo.both.{lang}.valid.tgt.txt'
    
    def _write_lines(target, lines):
        with open(target, 'w', encoding='utf-8') as f:
            f.writelines(lines)

    def _build_file():
        with open(pred_path, 'r', encoding='utf-8') as f:
            line_no = 0
            java_lines = []
            js_lines = []
            py_lines = []
            php_lines = []
            for line in f:
                line_no += 1
                if line_no <= 7000:
                    java_lines.append(line)
                    continue
                if line_no <= 14000:
                    js_lines.append(line)
                    continue
                if line_no <= 21000:
                    py_lines.append(line)
                    continue
                php_lines.append(line)
            _write_lines(pred_path.replace('.valid', '.java.valid'), java_lines)
            _write_lines(pred_path.replace('.valid', '.javascript.valid'), js_lines)
            _write_lines(pred_path.replace('.valid', '.python.valid'), py_lines)
            _write_lines(pred_path.replace('.valid', '.php.valid'), php_lines)
    
    _build_file()
    for lang in languages:
        print(f'------------{lang}-----------')
        running_local(pred_path.replace('.valid', f'.{lang}.valid'), gold_path.format(lang=lang))
                

if __name__ == '__main__':
    # pred_path = 'data/onmt/pred/lstm_cc.eighth.php.valid.pred.txt'
    # gold_path = 'data/onmt/php.both.valid.tgt.txt'
    # running_local(pred_path, gold_path)
    cal_four_lang()


'''53800
bleu:
0.21465210428879855
rouge:
(0.3662713569125462, {'rouge-1': {'f': 0.45229914595185877, 'p': 0.49377344781359955, 'r': 0.43984865448031774}, 'rouge-2': {'f': 0.21453223679029637, 'p': 0.23416711019322545, 'r': 0.20925626972233946}, 'rouge-l': {'f': 0.4319826879954835, 'p': 0.4673017407386495, 'r': 0.420334758106872}})

24400
bleu:
0.21682935701037556
rouge:
(0.3701591449406236, {'rouge-1': {'f': 0.45479782165292887, 'p': 0.49573022434659814, 'r': 0.44314881912788695}, 'rouge-2': {'f': 0.21908400942850764, 'p': 0.23842070615121014, 'r': 0.21466085575298305}, 'rouge-l': {'f': 0.43659560374043416, 'p': 0.4719121306247497, 'r': 0.42521923218127317}})
'''
