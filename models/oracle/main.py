'''
Author WHU ZFJ 2021
implementation of our oracle model with beam seach
'''
import random
import json
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge


random.seed(2021)
WRITE_BATCH = 10
BEAM_SIZE = 20
MAX_SEARCHING = 100


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


def construct_unigrams(content, reference):
    encoded_input = convention_tokenize(content)
    encoded_ref = convention_tokenize(reference)
    unigrams = set(encoded_input) & set(encoded_ref)
    return unigrams


def handle_special_characters(text):
    tokens = convention_tokenize(text)
    text_len = len(tokens)
    text = ' '.join(tokens)
    if not text.replace('.', '').strip():
        return text, 0
    return text, text_len


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
    avg_f = rouge1_f + rouge2_f + rougeL_f
    return avg_f/3


def calculate_score(reference, candidate_tokens):
    '''
    reference: str
    candidate_tokens: list(token1, token2, )
    '''
    reference, r_len = handle_special_characters(reference)
    candidate = ' '.join(candidate_tokens)
    candidate, c_len = handle_special_characters(candidate)
    if not c_len:
        bleu_score = 0
    else:
        bleu_score = nltk_bleu({0: [candidate]}, {0: [reference]})
    if not c_len:
        rouge_score = 0
    else:
        rouge_score = average_rouge({0: [candidate]}, {0: [reference]})
    avg = rouge_score + bleu_score
    return avg


def conditional_insert(reference, old_candidates, candidate_tokens):
    '''
    given raw candidates and new candidate
    new candidates will be generated
    reference: str
    candidate_tokens: list(token1, token2, )
    '''
    score = calculate_score(reference, candidate_tokens)
    old_candidates.insert(0, [candidate_tokens, score])
    old_candidates.sort(key=lambda x: -x[1])
    return old_candidates[:BEAM_SIZE]


def better_candidates(unigrams, old_candidates, reference):
    '''
    update (insert a token by left/right to the old one)
    unigrams:set()
    reference:str
    old_candidates:list([list(token1, token2,), score])
    '''
    ending = False
    result_candidates = old_candidates.copy()
    for candidate in old_candidates:
        candidate_tokens = candidate[0]
        for unigram in unigrams:
            new_candidate_tokens = candidate_tokens.copy()
            new_candidate_tokens.append(unigram)
            result_candidates = conditional_insert(
                                    reference,
                                    result_candidates,
                                    new_candidate_tokens)
    if result_candidates == old_candidates:
        ending = True
    return ending, result_candidates


def line_prediction(content, reference):
    '''
    generate the best candidate with given content
    content:str
    reference:str
    '''
    candidates = [[list(), 0]]  # (content, score)
    unigrams = construct_unigrams(content, reference)
    count = 0
    while True:
        # add a new unigram to each candidate
        ending, candidates = better_candidates(unigrams, candidates, reference)
        count += 1
        if ending:
            break
        if count > MAX_SEARCHING:
            break
    candidates.sort(key=lambda x: -x[1])
    rank_first = [' '.join(candidates[0][0]), candidates[0][1]]
    return rank_first


def write_lines(file_path, content):
    with open(file_path, 'a', encoding='utf-8') as f:
        f.writelines(content)


def file_prediction(source_path, pred_path, reference_path, total_line_count):
    '''
    given source_file (used for ccbert), generate candidate prediction
    '''
    result = []
    references = []
    with open(source_path, 'r', encoding='utf-8') as f:
        t = tqdm(total=total_line_count)
        for line in f:
            t.update(1)
            json_line = json.loads(line.strip())
            content = ' '.join(json_line['source_tokens'])
            reference = ' '.join(json_line['target_tokens'])
            result.append('{}\n'.format(
                line_prediction(content, reference)[0]))
            references.append('{}\n'.format(
                handle_special_characters(reference)[0]))
            if t.n % WRITE_BATCH == 0:
                write_lines(pred_path, result)
                write_lines(reference_path, references)
                result.clear()
                references.clear()
        t.close()
    if result:
        write_lines(pred_path, result)
        write_lines(reference_path, references)


if __name__ == '__main__':
    source_path = '../data/python.both.valid.jsonl'
    line_count = 2000
    pred_path = 'data/python.valid.pred.txt'
    reference_path = 'data/python.valid.ref.txt'
    file_prediction(source_path, pred_path, reference_path, line_count)
