'''
Author WHU ZFJ 2021
Filters and Cleaners to make datasets
'''
import json
import ipdb
import xmltodict
from tqdm import tqdm
from datetime import datetime

from data_tools.html_parser import BodyParser
from data_tools.utils import line_counter, write_to_file, BATCH_SIZE, \
    convention_tokenize


ALL_PROGRAM_LANGUAGES = set(['<javascript>', '<c#>', '<php>',
                            '<html>', '<c++>', '<python>', '<java>'])
TOP_15_LANGUAGES = set([
    '<javascript>', '<python>', '<java>', '<c#>', '<php>',
    '<html>', '<c++>', '<sql>', '<r>', '<c>',
    '<swift>', '<objective-c>', '<ruby>', '<vba>', '<typescript>',
])
TOP_10_LANGUAGES = set([
    '<javascript>', '<python>', '<java>', '<c#>', '<php>',
    '<c++>', '<c>', '<swift>', '<objective-c>', '<ruby>', '<vba>',
])
INTERROGATIVES = ['how', 'what', 'why', 'which', 'when']


def pick_and_clean_good_questions(source_xml_path, target_jsonl_path):
    '''
    input: xml file of questions to be filtered
    output: jsonl file containing parsed questions
    filter on the following conditions:
        1. scored more than 1
        2. get a accepted answer
        3. not closed yet
    '''
    total_line_count = line_counter(source_xml_path)
    handled_lines = []

    def _condition(x):
        result = True
        result = result and int(x['@Score']) > 1
        result = result and '@AcceptedAnswerId' in x
        result = result and '@ClosedDate' not in x
        return result
    with open(source_xml_path, 'r', encoding='utf-8') as f:
        t = tqdm(total=total_line_count)
        for line in f:
            t.update(1)
            try:
                line_json = xmltodict.parse(line)
            except Exception:
                continue
            if not _condition(line_json['row']):
                continue
            body_str = line_json['row']['@Body']
            parser = BodyParser()
            parser.feed(body_str)
            line_parsed = parser.get_result()
            line_json['row']['@Body'] = line_parsed
            raw_title = line_json['row']['@Title']
            line_json['row']['@Title'] = parser.denoising(raw_title)
            line_json_str = json.dumps(line_json['row'])
            handled_lines.append(f'{line_json_str}\n')
            if len(handled_lines) == BATCH_SIZE:
                write_to_file(handled_lines, target_jsonl_path, 'a')
                handled_lines.clear()
        t.close()
        if handled_lines:
            write_to_file(handled_lines, target_jsonl_path, 'a')


def pick_questions_by_time(source_jsonl_path, target_jsonl_path, years):
    '''
    input: josnl file of questions
    output: jsonl file of filtered questions
    filter out the questions posted in specific years
    '''
    total_len = line_counter(source_jsonl_path)
    filtered_lines = []
    t = tqdm(total=total_len)
    with open(source_jsonl_path, 'r', encoding='utf-8') as f_lines:
        for line in f_lines:
            t.update(1)
            line_js = json.loads(line.strip())
            time_str = line_js['@CreationDate']
            time_obj = datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S.%f')
            year = time_obj.strftime('%Y')
            if year in years:
                filtered_lines.append(line)
        t.close()
        write_to_file(filtered_lines, target_jsonl_path)


def pick_questions_by_language(language, source_jsonl_path, target_jsonl_path):
    '''
    input: josnl file of questions
    output: jsonl file of target language
    filter out the questions related to target programming language
    '''
    picked_lines = []
    current_language_tag = f'<{language}>'
    other_language_tags = ALL_PROGRAM_LANGUAGES - set([current_language_tag])
    with open(source_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            language_tag_flag = False
            line_json = json.loads(line.strip())
            tags_raw = line_json['@Tags']
            for lang_tag in other_language_tags:
                if lang_tag in tags_raw:
                    language_tag_flag = True
            if language_tag_flag:
                continue
            if current_language_tag in tags_raw:
                picked_lines.append(line)
        write_to_file(picked_lines, target_jsonl_path)


def pick_questions_by_languages(languages, source_jsonl_path, target_jsonl_path):
    '''
    input: josnl file of questions
    output: jsonl file of target language
    filter out the questions related to target programming language
    '''
    picked_lines = []
    with open(source_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            to_pick = False
            line_json = json.loads(line.strip())
            tags_raw = line_json['@Tags']
            for lang_tag in languages:
                if lang_tag in tags_raw:
                    to_pick = True
            if not to_pick:
                continue
            picked_lines.append(line)
        write_to_file(picked_lines, target_jsonl_path)


def pick_questions_by_body_content(
            source_jsonl_path,
            target_jsonl_path,
            check_bi_modal=True,
            check_length=True,
            check_interrogative=True
        ):
    '''
    input: josnl file of questions
    output: jsonl file of filtered questions
    filter on the following conditions:
        1. containing both text block and code block
        2. have a body length less than 1000 and a title less than 25
        3. have one of "how, what, why, which, when" in the title
    '''
    count_no_both_text_and_code = 0
    count_length_too_long = 0
    count_no_interrogative_words = 0
    total_line_count = line_counter(source_jsonl_path)
    filtered_lines = []
    with open(source_jsonl_path, 'r', encoding='utf-8') as f:
        t = tqdm(total=total_line_count)
        for line in f:
            t.update(1)
            line_json = json.loads(line.strip())
            if check_bi_modal:
                # skip those not containing both text and code
                line_body = line_json['@Body']
                tags = [item[0] for item in line_body]
                if ('code' not in tags) or ('text' not in tags):
                    count_no_both_text_and_code += 1
                    continue
            if check_length:
                # skip those whose body length exceeds 1000
                body_len = 0
                for segment in line_json['@Body']:
                    body_len += len(convention_tokenize(segment[1]))
                title_len = len(convention_tokenize(line_json['@Title']))
                if body_len > 1000 or title_len > 25:
                    count_length_too_long += 1
                    continue
            if check_interrogative:
                # skip those whose tile doesnt have interrogative words
                interrogative_in_title = False
                title = line_json['@Title'].lower()
                for i in INTERROGATIVES:
                    if i in title:
                        interrogative_in_title = True
                if not interrogative_in_title:
                    count_no_interrogative_words += 1
                    continue
            filtered_lines.append(line)
        t.close()
    write_to_file(filtered_lines, target_jsonl_path)
    print(f'total raw lines {total_line_count}')
    print(f'lines not with both text and code {count_no_both_text_and_code}')
    print(f'lines with a body longer than 1000 {count_length_too_long}')
    print(f'not contain interrogative words {count_no_interrogative_words}')
    print(f'lines filtered {len(filtered_lines)}')
