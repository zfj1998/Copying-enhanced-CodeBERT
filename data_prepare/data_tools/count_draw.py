'''
Author WHU ZFJ 2021
Statistics jobs of our collected data
'''
import json
from unicodedata import decimal
import numpy as np
import ipdb
from tqdm import tqdm
from datetime import datetime
from collections import Counter, OrderedDict
from pylab import subplots_adjust
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter

from data_tools.utils import line_counter, convention_tokenize


class OrderedCounter(Counter, OrderedDict):
    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, OrderedDict(self))


def calc_ratio_of_bimodal_data(source_jsonl_path):
    '''
    input: jsonl file containing questions to analyze
    count questions that contain text or code in year
    '''
    result = []
    years = set()
    with_code_key = '{year}-with-code'
    with_text_key = '{year}-with-text'
    total_len = line_counter(source_jsonl_path)
    t = tqdm(total=total_len)
    with open(source_jsonl_path, 'r', encoding='utf-8') as f_lines:
        for line in f_lines:
            t.update(1)
            line_js = json.loads(line.strip())
            time_str = line_js['@CreationDate']
            time_obj = datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S.%f')
            year = time_obj.strftime('%Y')
            years.add(year)
            result.append(year)
            tags = [item[0] for item in line_js['@Body']]
            if 'code' in tags:
                result.append(with_code_key.format(year=year))
            if 'text' in tags:
                result.append(with_text_key.format(year=year))
        t.close()
    counter = OrderedCounter(result)
    for year in sorted(list(years)):
        year_total = counter[year]
        year_text_count = counter[with_text_key.format(year=year)]
        year_code_count = counter[with_code_key.format(year=year)]
        counter[f'{year}-text-ratio'] = round(year_text_count / year_total, 4)
        counter[f'{year}-code-ratio'] = round(year_code_count / year_total, 4)
    return counter


def multi_Y_line_chart_inner(time_array, text_ratio, code_ratio, save_path):
    '''
    make chart with double lines
    '''
    fig, ax = plt.subplots(dpi=500)
    fig.subplots_adjust(right=0.75)  # adjust image width
    twin1 = ax.twinx()

    p1, = ax.plot(time_array, text_ratio, "blue",  # dodgerblue
                  label='Questions With Text Description')
    p2, = twin1.plot(time_array, code_ratio, "red",  # tomato
                     label='Questions With Code Snippet')

    # adjust range of Y axis
    ax.set_ylim(0.3, 1.1)
    twin1.set_ylim(0.3, 1.1)

    ax.set_ylabel('Propotion')
    twin1.set_ylabel('Propotion')

    # make line's color the same as axis
    ax.yaxis.label.set_color(p1.get_color())
    twin1.yaxis.label.set_color(p2.get_color())
    tkw = dict(size=4, width=1.5)  # change the size of axis marks
    ax.tick_params(axis='y', colors=p1.get_color(), **tkw)
    twin1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    ax.tick_params(axis='x', **tkw, rotation=45)
    ax.legend(handles=[p1, p2])
    plt.savefig(save_path)


def multi_Y_line_chart(data, save_path):
    '''
    receive data calculated by calc_ratio_of_bimodal_data
    make a multi lines chart
    '''
    time_array = []
    text_ratio = []
    code_ratio = []
    for year in range(2008, 2021):
        year = str(year)
        time_array.append(year)
        with_text = f'{year}-text-ratio'
        with_code = f'{year}-code-ratio'
        text_ratio.append(data[with_text])
        code_ratio.append(data[with_code])
    multi_Y_line_chart_inner(time_array, text_ratio, code_ratio, save_path)


def count_length(source_jsonl_path):
    '''
    count length of question body and question title
    tokenization is done by our algorithm
    '''
    with open(source_jsonl_path, mode='r', encoding='utf-8') as f:
        title_length = []
        body_length = []
        text_length = []
        code_length = []
        for line in f:
            js_line = json.loads(line)
            body_content = js_line['@Body']
            title_tokens = convention_tokenize(js_line['@Title'])
            body_tokens = []
            code_tokens = []
            text_tokens = []
            for segment in body_content:
                body_tokens += convention_tokenize(segment[1])
                if segment[0] == 'code':
                    code_tokens += convention_tokenize(segment[1])
                if segment[0] == 'text':
                    text_tokens += convention_tokenize(segment[1])
            title_length.append(len(title_tokens))
            body_length.append(len(body_tokens))
            text_length.append(len(code_tokens))
            code_length.append(len(text_tokens))
    title_length = np.array(title_length)
    body_length = np.array(body_length)
    code_length = np.array(code_length)
    text_length = np.array(text_length)
    return title_length, body_length, code_length, text_length


def count_avg_length(source_jsonl_path):
    '''
    count average length of question body and question title
    '''
    title_length, body_length, code_length, text_length = count_length(source_jsonl_path)
    print(f'title avg len: {title_length.mean()}')
    print(f'body avg len: {body_length.mean()}')
    print(f'code avg len: {code_length.mean()}')
    print(f'text avg len: {text_length.mean()}')


def calc_token_overlap(source_jsonl_path):
    '''
    calculate the overlap bewteen tokens in title and tokens in bi-modal data
    '''
    total_len = line_counter(source_jsonl_path)
    t = tqdm(total=total_len)
    with open(source_jsonl_path, mode='r', encoding='utf-8') as f_lines:
        token_in_code_ratio = []
        token_in_text_ratio = []
        for line in f_lines:
            t.update(1)
            line_js = json.loads(line.strip())
            title = line_js['@Title']
            title_tokens = set(convention_tokenize(title))
            body = line_js['@Body']
            body_text = ''
            body_code = ''
            for tag, content in body:
                if tag == 'text':
                    body_text += f' {content}'
                elif tag == 'code':
                    body_code += f' {content}'
            body_text_tokens = set(convention_tokenize(body_text))
            body_code_tokens = set(convention_tokenize(body_code))
            title_in_text_ratio = len(title_tokens & body_text_tokens) \
                / len(title_tokens)
            title_in_code_ratio = len(title_tokens & body_code_tokens) \
                / len(title_tokens)
            token_in_text_ratio.append(title_in_text_ratio)
            token_in_code_ratio.append(title_in_code_ratio)
        t.close()
        result = {
            'Overlap Between Title and Code Snippet': token_in_code_ratio,
            'Overlap Between Title and Text Description': token_in_text_ratio
        }
        return result


def plot_histogram(x, save_path, x_label, distance=0.1):
    '''
    x is a list containing numbers
    distance is an int number indicating the x-axis distance
    '''
    d = distance
    plt.figure(dpi=300)
    min_x = int(min(x))
    max_x = int(max(x))
    range_by_d = np.arange(min_x, max_x + d, d)
    # plt.hist(x, range_by_d, weights=np.ones(len(x))/len(x))
    plt.hist(x, bins=20, facecolor="dodgerblue", edgecolor="black", alpha=0.7, weights=np.ones(len(x)) / len(x))
    plt.xlabel(x_label)
    plt.ylabel('Probability')
    plt.xticks(range_by_d, rotation=45)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(decimals=2))
    # plt.grid()
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig(save_path)


def draw_token_overlap(data, save_path):
    '''
    use data calculated from calc_token_overlap to draw two histograms
    '''
    prompt = {
        0: 'Ratio of Title Tokens Appearing in Code Snippets',
        1: 'Ratio of Title Tokens Appearing in Text descriptions'
    }
    for index, key in enumerate(data.keys()):
        plot_histogram(data[key],
                       save_path.replace('.png', f'.{index}.png'),
                       prompt[index])


def calc_overlap_expectation(data):
    '''
    use data calculated from calc_token_overlap to find the math expectation
    '''
    for key in data.keys():
        counter = Counter(data[key])
        total_counts = len(data[key])
        expectation = 0
        for ratio in counter.keys():
            expectation += counter[ratio] / total_counts * ratio
        print(f'{key} expectation {expectation}')


def draw_length_distribution(py_body, py_title, ja_body, ja_title, save_path):
    '''
    use data calculated by count_length to show token length distribution
    '''
    body_len = [py_body, ja_body]
    title_len = [py_title, ja_title]
    labels = ['Python', 'Java']
    plt.figure(dpi=300)
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
    # rectangular box plot
    bplot1 = ax1.boxplot(body_len,
                         sym='',
                         vert=True,  # vertical box alignment
                         patch_artist=True,  # fill with color
                         labels=labels)  # will be used to label x-ticks
    # notch shape box plot
    bplot2 = ax2.boxplot(title_len,
                         sym='',
                         vert=True,  # vertical box alignment
                         patch_artist=True,  # fill with color
                         labels=labels)  # will be used to label x-ticks
    # fill with colors
    colors = ['pink', 'lightblue']
    for bplot in (bplot1, bplot2):
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
    # adding horizontal grid lines
    for ax in [ax1, ax2]:
        # ax.yaxis.grid(True)
        ax.set_ylabel('Token Count')
    ax1.set_xlabel('Entire Body')
    ax2.set_xlabel('Code Snippets')
    plt.gcf().subplots_adjust(bottom=0.15, wspace=0.3)
    plt.savefig(save_path)


def count_questions_by_year(source_jsonl_path):
    count = {}
    for year in range(2008, 2021):
        count[year] = 0
    with open(source_jsonl_path, mode='r', encoding='utf-8') as f:
        for line in f:
            line_js = json.loads(line.strip())
            time_str = line_js['@CreationDate']
            time_obj = datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S.%f')
            year = time_obj.strftime('%Y')
            count[int(year)] += 1
    result = [count[year] for year in range(2008, 2021)]
    return result


def draw_year_distribution(py_questions, ja_questions, save_path):
    '''
    make a bar chart describing the year distribution of questions
    '''
    labels = [str(year) for year in range(2008, 2021)]
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots(dpi=500)
    rects1 = ax.bar(x - width/2, py_questions, width, label='Python', color='lightcoral', edgecolor='black')
    rects2 = ax.bar(x + width/2, ja_questions, width, label='Java', color='lightskyblue', edgecolor='black')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Count')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    plt.xticks(rotation=45)
    ax.legend()
    fig.tight_layout()
    plt.savefig(save_path)
