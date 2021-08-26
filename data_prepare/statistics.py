'''
Author WHU ZFJ 2021
Main entry for statistics of our paper
'''
from data_tools.filter_clean import \
    pick_and_clean_good_questions, pick_questions_by_time
from data_tools.utils import line_counter, save_cache, load_cache
from data_tools.count_draw import \
    calc_ratio_of_bimodal_data, multi_Y_line_chart, count_avg_length, \
    calc_token_overlap, draw_token_overlap, calc_overlap_expectation


def both_text_and_code_matters():
    '''
    figure out how many questions contain code and text
    '''
    xml_of_all = 'data/xml/Posts.xml'
    jsonl_of_good_questions = 'data/json/Good_questions.jsonl'
    bimodal_ratio_line_chart = 'charts/bimodal_ratio_line_chart.png'
    good_questions_from_2019_to_2020 = \
        'data/json/Good_questions_from_2019_to_2020.jsonl'
    # pick_and_clean_good_questions(xml_of_all, jsonl_of_good_questions)

    def _describe_good_questions():
        line_counter(jsonl_of_good_questions)
        # ratio = calc_ratio_of_bimodal_data(jsonl_of_good_questions)
        cache_path = 'data/cache/ratio_good_questions.pkl'
        # save_cache(ratio, cache_path)
        ratio = load_cache(cache_path)
        multi_Y_line_chart(ratio, bimodal_ratio_line_chart)
    # _describe_good_questions()
    # pick_questions_by_time(jsonl_of_good_questions,
    #                        good_questions_from_2019_to_2020, ['2019', '2020'])
    # count_avg_length(good_questions_from_2019_to_2020)

    def _describe_importance_of_bi_modal_data():
        # overlap = calc_token_overlap(good_questions_from_2019_to_2020)
        cache_path = 'data/cache/overlap_2019_2020.pkl'
        # save_cache(overlap, cache_path)
        overlap = load_cache(cache_path)
        # draw_token_overlap(overlap, 'charts/bimodal_overlap_2.png')
        calc_overlap_expectation(overlap)
    _describe_importance_of_bi_modal_data()


if __name__ == '__main__':
    both_text_and_code_matters()
