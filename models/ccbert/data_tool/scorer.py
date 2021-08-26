'''
Author WHU ZFJ 2021
used for scoring
'''
from rouge import Rouge


special_tokens_id = list(range(33, 48))
special_tokens_id += list(range(58, 65))
special_tokens_id += list(range(91, 97))
special_tokens_id += list(range(123, 127))
special_tokens = [chr(i) for i in special_tokens_id]

def convention_tokenize(text):
    '''
    Word segmentation for special symbols 
    '''
    for st in special_tokens:
        text = f' {st} '.join(text.split(st)).strip()
    tokens = text.split()
    return tokens

def rouge_from_maps(refs, hyps):
    '''
    Calculate the sum of the f-values of the three rouge scores in a weighted manner
    The purpose is to compare with the training results of CodeBERT-text 
    return avg_score
    '''
    rouge = Rouge()
    # hyps/refs: list[[tokens], [tokens], [tokens]]
    hyps = [' '.join(tokens) for tokens in hyps]
    refs = [' '.join(tokens) for tokens in refs]
    # hyps/refs: list[str, str, str]
    try:
        scores = rouge.get_scores(hyps, refs, avg=True)
    except Exception as e:
        return 0
    rouge1_f = scores['rouge-1']['f']
    rouge2_f = scores['rouge-2']['f']
    rougeL_f = scores['rouge-l']['f']
    avg_f = 0.2*rouge1_f + 0.3*rouge2_f + 0.5*rougeL_f
    return avg_f