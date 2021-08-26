'''
Author WHU ZFJ 2021
Main entry for dataset construction
'''
from data_tools.filter_clean import pick_questions_by_language, \
    pick_questions_by_body_content
from data_tools.count_draw import draw_length_distribution, count_length, \
    count_questions_by_year, draw_year_distribution
from data_tools.divider import partition_on_time


def build_python_and_java_datasets():
    '''
    extract and clean data for our datasets
    '''
    jsonl_of_good_questions = 'data/json/Good_questions.jsonl'
    python_questions = 'data/json/Good_python_questions.jsonl'
    java_questions = 'data/json/Good_java_questions.jsonl'
    filtered_python = 'data/json/filtered_good_python_questions.jsonl'
    filtered_java = 'data/json/filtered_good_java_questions.jsonl'
    pick_questions_by_language('python', jsonl_of_good_questions,
                               python_questions)
    pick_questions_by_language('java', jsonl_of_good_questions,
                               java_questions)
    pick_questions_by_body_content(python_questions, filtered_python)
    '''
    total raw lines 260229
    lines not with both text and code 26820
    lines with a body longer than 1000 12086
    not contain interrogative words 156865
    lines filtered 64458
    '''
    pick_questions_by_body_content(java_questions, filtered_java)
    '''
    total raw lines 267375
    lines not with both text and code 51774
    lines with a body longer than 1000 18158
    not contain interrogative words 136325
    lines filtered 61118
    '''


def describe_datasets():
    '''
    draw charts and tables that make our datasets clear
    '''
    filtered_python = 'data/json/filtered_good_python_questions.jsonl'
    filtered_java = 'data/json/filtered_good_java_questions.jsonl'
    length_distribution = 'charts/length_distribution2.png'
    year_distribution = 'charts/year_distribution.png'
    py_title, py_body, py_code, py_text = count_length(filtered_python)
    ja_title, ja_body, ja_code, ja_text = count_length(filtered_java)
    draw_length_distribution(py_body, py_code, ja_body,
                             ja_code, length_distribution)
    python_year_count = count_questions_by_year(filtered_python)
    java_year_count = count_questions_by_year(filtered_java)
    draw_year_distribution(python_year_count, java_year_count,
                           year_distribution)


def dataset_partition():
    python_source = 'data/json/filtered_good_python_questions.jsonl'
    java_source = 'data/json/filtered_good_java_questions.jsonl'
    partition_on_time(python_source)
    partition_on_time(java_source)


if __name__ == '__main__':
    # build_python_and_java_datasets()
    describe_datasets()
    # dataset_partition()
