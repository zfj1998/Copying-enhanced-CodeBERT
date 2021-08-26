'''
Author WHU ZFJ 2021
Score with BLEU, Rouge
Scored with lower case, tokenize in our own way
'''
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from meteor import Meteor


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



def running_local():
    pred_path = 'data/...'
    gold_path = 'data/...'
    hypotheses, references = get_word_maps_from_file(pred_path,
                                                     gold_path, True)
    bleu = nltk_bleu(hypotheses, references)
    rouge_score = average_rouge(hypotheses, references)
    print(f'bleu:\n{bleu}')
    print(f'rouge:\n{rouge_score}')


if __name__ == '__main__':
    running_local()
