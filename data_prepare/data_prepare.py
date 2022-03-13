'''
Author WHU ZFJ 2021
Main entry for dataset construction
'''
from data_tools.filter_clean import pick_questions_by_language, \
    pick_questions_by_body_content, pick_questions_by_time
from data_tools.count_draw import draw_length_distribution_by_language, count_length, \
    count_questions_by_year, draw_year_distribution
from data_tools.divider import partition_on_time, mix_datasets


def build_datasets_for_four_languages():
    '''
    extract and clean data for our datasets
    '''
    jsonl_of_good_questions = 'data/json/Good_questions.jsonl'
    php_questions = 'data/json/Good_php_questions.jsonl'
    javascript_questions = 'data/json/Good_javascript_questions.jsonl'
    filtered_php = 'data/json/w_wo_interrogative_php_questions.jsonl'
    filtered_javascript = 'data/json/w_wo_interrogative_javascript_questions.jsonl'
    # pick_questions_by_language('php', jsonl_of_good_questions,
    #                            php_questions)
    # pick_questions_by_language('javascript', jsonl_of_good_questions,
    #                            javascript_questions)
    # pick_questions_by_body_content(php_questions, filtered_php, check_interrogative=False)
    '''
    total raw lines 142571
    lines not with both text and code 24156(0.17)
    lines with a body longer than 1000 6188
    not contain interrogative words 83689(0.58)
    lines filtered 28538
    '''
    # pick_questions_by_body_content(javascript_questions, filtered_javascript, check_interrogative=False)
    '''
    total raw lines 225255
    lines not with both text and code 28311(0.12)
    lines with a body longer than 1000 8570
    not contain interrogative words 130666(0.58)
    lines filtered 57708
    '''
    python_questions = 'data/json/Good_python_questions.jsonl'
    java_questions = 'data/json/Good_java_questions.jsonl'
    filtered_python = 'data/json/w_wo_interrogative_python_questions.jsonl'
    filtered_java = 'data/json/w_wo_interrogative_java_questions.jsonl'
    pick_questions_by_language('python', jsonl_of_good_questions,
                               python_questions)
    pick_questions_by_language('java', jsonl_of_good_questions,
                               java_questions)
    pick_questions_by_body_content(python_questions, filtered_python, check_interrogative=False)
    '''
    total raw lines 260229
    lines not with both text and code 26820(0.1)
    lines with a body longer than 1000 12086
    not contain interrogative words 156865(0.60)
    lines filtered 64458
    '''
    pick_questions_by_body_content(java_questions, filtered_java, check_interrogative=False)
    '''
    total raw lines 267375
    lines not with both text and code 51774(0.19)
    lines with a body longer than 1000 18158
    not contain interrogative words 136325(0.51)
    lines filtered 61118
    '''
    # ruby_questions = 'data/json/Good_ruby_questions.jsonl'
    # go_questions = 'data/json/Good_go_questions.jsonl'
    # filtered_ruby = 'data/json/filtered_good_ruby_questions.jsonl'
    # filtered_go = 'data/json/filtered_good_go_questions.jsonl'
    # pick_questions_by_language('ruby', jsonl_of_good_questions,
    #                            ruby_questions)
    # pick_questions_by_language('go', jsonl_of_good_questions,
    #                            go_questions)
    # pick_questions_by_body_content(ruby_questions, filtered_ruby)
    # '''
    # total raw lines 42480
    # lines not with both text and code 6205
    # lines with a body longer than 1000 1867
    # not contain interrogative words 23057
    # lines filtered 11351
    # '''
    # pick_questions_by_body_content(go_questions, filtered_go)
    # '''
    # total raw lines 11433
    # lines not with both text and code 1259
    # lines with a body longer than 1000 289
    # not contain interrogative words 6400
    # lines filtered 3485
    # '''

def dataset_partition():
    php_source = 'data/json/w_wo_interrogative_php_questions.jsonl'
    javascript_source = 'data/json/w_wo_interrogative_javascript_questions.jsonl'
    partition_on_time(php_source, 8000)
    partition_on_time(javascript_source, 14000)
    python_source = 'data/json/w_wo_interrogative_python_questions.jsonl'
    java_source = 'data/json/w_wo_interrogative_java_questions.jsonl'
    partition_on_time(python_source, 14000)
    partition_on_time(java_source, 14000)

def dataset_of_four_languages(partition=1):
    # partition is used to control the size of the dataset for investigating data-hungry situation
    source_paths = []
    for lang in ['java', 'javascript', 'python', 'php']:
        source_paths.append(f'data/json/w_wo_interrogative_{lang}_questions.jsonl')
    target_path = 'data/json/w_wo_interrogative_four_lang_questions.jsonl'
    mix_datasets(source_paths, target_path, partition)

def dataset_of_bi_modal():
    jsonl_of_good_questions = 'data/json/Good_questions.jsonl'
    jsonl_of_bi_modal_good_questions = 'data/json/Good_bi_modal_questions.jsonl'
    pick_questions_by_body_content(jsonl_of_good_questions, jsonl_of_bi_modal_good_questions, check_length=False, check_interrogative=False)


if __name__ == '__main__':
    # build_datasets_for_four_languages()
    # dataset_partition()
    dataset_of_four_languages()
    # dataset_of_bi_modal()
