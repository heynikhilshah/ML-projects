from bs4 import BeautifulSoup
import os
import re
import json
import requests


def html_parser():
    list_html = []
    for season in range(8):
        url = 'https://genius.com/albums/Game-of-thrones/Season-' + str(season + 1) + '-scripts'
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        a = soup.findAll("a", {"class": "u-display_block"})
        list_links = [data['href'] for data in a]
        for key, link in enumerate(list_links):
            url = link
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            a = soup.findAll("div", {"class": "lyrics"})
            list_html.append(a)
    return list_html


script_list = html_parser()

script_list.pop(40)
script_list.pop(51)


def scene_to_line_dict(list_html):
    r = re.compile("^\w+:|\w+\s\w+:")
    lines_dict = {}
    n = 0
    for e in range(73):

        if e == 6:
            for i in list(list_html[e][0].findAll("p")[1:]):
                text = i.text.split('\n')
                n = n + 1
                lines = []
                for line in text:
                    if re.match(r, line):
                        lines.append(line)
                    else:
                        pass
                    if lines != []:
                        lines_dict[n] = lines


        elif e == 7:
            for p, i in enumerate(list(list_html[e][0].findAll("p")[1:])):
                text = i.text.split('\n')
                n = n + 1
                lines = []
                for line in text:
                    if (line.startswith('Back')):
                        n = n + 1
                        lines = []
                        continue
                    elif re.match(r, line):
                        lines.append(line)
                    else:
                        pass
                    if lines != []:
                        lines_dict[n] = lines

        elif e == 38:
            for p, i in enumerate(list(list_html[e][0].findAll("p")[1:])):
                text = i.text.split('\n')
                n = n + 1
                lines = []
                for line in text:
                    if (line.startswith('[')):
                        n = n + 1
                        lines = []
                        continue
                    elif re.match(r, line):
                        lines.append(line)
                    else:
                        pass
                    if lines != []:
                        lines_dict[n] = lines

        else:
            for p, i in enumerate(list(list_html[e][0].findAll("p")[1:])):
                text = i.text.split('\n')
                n = n + 1
                lines = []
                for line in text:
                    if (re.search('scene', line) or re.search('Scene', line) or re.match('EPISODE', line)
                            or re.match('-', line) or re.match('CUT TO:', line) or re.match('INT:', line)
                            or re.match('EXT:', line) or re.match('EXT.', line) or re.match('INT.', line)):

                        n = n + 1
                        lines = []
                        continue
                    elif re.match(r, line):
                        lines.append(line)
                    else:
                        pass
                    if lines != []:
                        lines_dict[n] = lines
    with open('full_script.json', 'w') as f:
        json.dump(lines_dict, f)


scene_to_line_dict(script_list)
