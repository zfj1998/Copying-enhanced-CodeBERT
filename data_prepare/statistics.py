'''
Author WHU ZFJ 2021
Main entry for statistics of our paper
'''
from functools import cache
import ipdb
from data_tools.filter_clean import \
    pick_and_clean_good_questions, pick_questions_by_time, pick_questions_by_languages, TOP_10_LANGUAGES
from data_tools.utils import line_counter, save_cache, load_cache
from data_tools.count_draw import \
    calc_ratio_of_bimodal_data, multi_Y_line_chart, count_avg_length, \
    calc_token_overlap, draw_token_overlap, calc_overlap_expectation, count_length, \
    draw_length_distribution_by_language, draw_year_distribution, draw_length_distribution, \
    count_questions_by_year, draw_token_overlap_all_years


def both_text_and_code_matters():
    '''
    figure out how many questions contain code and text
    '''
    xml_of_all = 'data/xml/Posts.xml'
    jsonl_of_good_questions = 'data/json/Good_questions.jsonl'
    bimodal_ratio_line_chart = 'charts/bimodal_ratio_line_chart_10.png'
    bi_modal_good_questions = \
        'data/json/Good_bi_modal_questions.jsonl' # 2008-2020 所有的双模态问题
    jsonl_of_good_questions_with_15language_tags = \
        'data/json/Good_questions_15_langs.jsonl'
    jsonl_of_good_questions_with_10language_tags = \
        'data/json/Good_questions_10_langs.jsonl'
    bi_model_good_questions_by_year = 'data/json/Good_bi_modal_questions_{year}.jsonl'

    # pick_and_clean_good_questions(xml_of_all, jsonl_of_good_questions)

    def _describe_good_questions():
        # line_counter(jsonl_of_good_questions_with_10language_tags)
        ratio = calc_ratio_of_bimodal_data(jsonl_of_good_questions)
        # cache_path = 'data/cache/ratio_good_questions.pkl'
        # save_cache(ratio, cache_path)
        # ratio = load_cache(cache_path)
        multi_Y_line_chart(ratio, bimodal_ratio_line_chart)
    # _describe_good_questions()
    # pick_questions_by_time(jsonl_of_good_questions,
    #                        bi_modal_good_questions, ['2019', '2020'])
    # count_avg_length(bi_modal_good_questions)

    # pick_questions_by_languages(TOP_10_LANGUAGES, jsonl_of_good_questions_with_15language_tags, jsonl_of_good_questions_with_10language_tags)
    # _describe_good_questions()

    def _describe_importance_of_bi_modal_data():
        # overlap = calc_token_overlap(bi_modal_good_questions)
        cache_path = 'data/cache/overlap_bi_modal_good_questions.pkl'
        # save_cache(overlap, cache_path)
        overlap = load_cache(cache_path)
        # draw_token_overlap(overlap, 'charts/bimodal_overlap_2.png')
        calc_overlap_expectation(overlap)

    def _overlap_of_all_years():
        # overlaps = dict()
        # for year in range(2008, 2021):
        #     # pick_questions_by_time(bi_modal_good_questions, bi_model_good_questions_by_year.format(year=year), [str(year)])
        #     overlap, total_len = calc_token_overlap(bi_model_good_questions_by_year.format(year=year))
        #     expectations = calc_overlap_expectation(overlap)
        #     overlaps[year] = {
        #         'code_exp': expectations['Overlap Between Title and Code Snippet'],
        #         'text_exp': expectations['Overlap Between Title and Text Description'],
        #         'total_count': total_len
        #     }
        cache_path = 'data/cache/overlap_bi_modal_good_questions_all_years.pkl'
        # save_cache(overlaps, cache_path)
        overlaps = load_cache(cache_path)
        draw_token_overlap_all_years(overlaps, 'charts/bimodal_overlap_all_years.png')

    _overlap_of_all_years()


def describe_datasets():
    '''
    draw charts and tables that make our datasets clear
    '''
    # filtered_python = 'data/json/filtered_good_python_questions.jsonl'
    # filtered_java = 'data/json/filtered_good_java_questions.jsonl'
    # length_distribution = 'charts/length_distribution2.png'
    # year_distribution = 'charts/year_distribution.png'
    # py_title, py_body, py_code, py_text = count_length(filtered_python)
    # ja_title, ja_body, ja_code, ja_text = count_length(filtered_java)
    # draw_length_distribution_by_language(py_body, py_code, ja_body,
    #                                      ja_code, length_distribution)
    # python_year_count = count_questions_by_year(filtered_python)
    # java_year_count = count_questions_by_year(filtered_java)
    # draw_year_distribution(python_year_count, java_year_count,
    #                        year_distribution)
    filtered_data = 'data/json/Good_bi_modal_questions.jsonl'
    length_distribution = 'charts/length_distribution.png'
    cache_path = 'data/cache/length_distribution.pkl'
    # line_counter(filtered_data)
    # result = count_length(filtered_data)
    # save_cache(result, cache_path)
    result = load_cache(cache_path)
    title, body, code, text = result
    ipdb.set_trace()
    draw_length_distribution(body, code, text, length_distribution)

if __name__ == '__main__':
    both_text_and_code_matters()
