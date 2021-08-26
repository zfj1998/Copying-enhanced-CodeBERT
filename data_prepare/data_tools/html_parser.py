'''
Author WHU ZFJ 2021
A tool for parsing html in question bodies
and turn them into segements composed of text descriptions and code snippets
'''
import html
from html.parser import HTMLParser


class BodyParser(HTMLParser):
    def __init__(self):
        super(BodyParser, self).__init__()
        self.result = []
        self.current_tag = None
        self.current_text = ''
        self.current_code = ''

    def simplify_tag(self, tag):
        if tag == 'code':
            return tag
        return 'text'

    def handle_starttag(self, tag, attrs):
        tag = self.simplify_tag(tag)
        if not self.current_tag:
            self.current_tag = tag
            return
        if tag == 'code' and tag != self.current_tag:
            self.result.append(('text', self.current_text))
            self.current_text = ''
        elif tag == 'text' and tag != self.current_tag:
            self.result.append(('code', self.current_code))
            self.current_code = ''
        self.current_tag = tag

    def handle_endtag(self, tag):
        tag = self.simplify_tag(tag)
        if tag == 'code':
            self.result.append(('code', self.current_code))
            self.current_code = ''
            self.current_tag = 'text'  # cause no other tags in <code>
        elif tag == 'text':
            self.result.append(('text', self.current_text))
            self.current_text = ''

    def handle_data(self, data):
        data = html.unescape(data).strip()
        if not data:
            return
        if self.current_tag == 'text':
            self.current_text += f'{data} '
        elif self.current_tag == 'code':
            self.current_code += f'{data} '

    def denoising(self, data):
        # remove characters that are not in ASCII
        for char in ['\r\n', '\r', '\n']:
            data = data.replace(char, ' ')
        data = ''.join([i if ord(i) < 128 else ' ' for i in data])
        data = ' '.join(data.split())
        return data

    def get_result(self):
        '''
        return: [(tag, content),...]
        '''
        if not self.result:
            return []
        merged_result = []
        last_tag = self.result[0][0]
        current_content = self.result[0][1]
        for segment in self.result[1:]:
            tag, content = segment
            if (not content) or (not content.strip()):
                continue
            if tag != last_tag:
                merged_result.append((last_tag, current_content))
                last_tag = tag
                current_content = content
                continue
            current_content += content
        merged_result.append((last_tag, current_content))
        cleaned_result = []
        # remove items that are empty
        for item in merged_result:
            if not item[1]:
                continue
            denoised_content = self.denoising(item[1])
            if denoised_content:
                cleaned_result.append((item[0], denoised_content))
        return cleaned_result
