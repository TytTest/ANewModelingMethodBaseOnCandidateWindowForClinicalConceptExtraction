import sys

sys.path.append('./bert')
import os
import re
import tokenization
import tensorflow


def read_data(data_dir):
    text_dir = data_dir + '/text'
    concept_dir = data_dir + '/concept'
    file_list = os.listdir(text_dir)
    data_list = []
    for cur_file in file_list:
        file_name = cur_file[:-4]
        text_file_path = os.path.join(text_dir, file_name + '.txt')
        concept_file_path = os.path.join(concept_dir, file_name + '.con')
        if os.path.isfile(text_file_path) and os.path.isfile(concept_file_path):
            text_file = open(text_file_path, encoding='utf-8')
            concept_file = open(concept_file_path, encoding='utf-8')
            # print(file_name)
            # print('\n', concept_file_path)
            concept_list = []
            for line in concept_file:
                data = re.search('c="([\s\S]*)" (\d+):(\d+) (\d+):(\d+)', line)
                concept_list.append({
                    'concept_text': data.group(1),
                    'start_line': int(data.group(2)),
                    'start_loc': int(data.group(3)),
                    'end_line': int(data.group(4)),
                    'end_loc': int(data.group(5)),
                    'concept_type': re.search('t="([^"]*)"', line).group(1)})

            for i, line in enumerate(text_file):
                if line.startswith('\ufeff'):
                    line = line[1:]
                word_list = line[:-1].split(' ')
                concept_dict = {}
                for j, concept in enumerate(concept_list):
                    word_start_loc = 0
                    word_end_loc = 0
                    loc = -1
                    for k, word in enumerate(word_list):
                        if word != ' ' and word != '' and word != '\t' and word is not None:
                            loc += 1
                        if loc == concept['start_loc']:
                            word_start_loc = k
                        if loc == concept['end_loc']:
                            word_end_loc = k
                            break

                    if concept['start_line'] == i + 1 and concept['end_line'] == i + 1:
                        if ' '.join(word_list[word_start_loc: word_end_loc + 1]).lower() == concept['concept_text'].lower().strip():
                            start_loc = 0
                            for k in range(word_start_loc):
                                start_loc += len(word_list[k])
                                start_loc += 1
                            end_loc = 0
                            for k in range(word_end_loc):
                                end_loc += len(word_list[k])
                                end_loc += 1
                            end_loc += len(word_list[word_end_loc])

                            if line[start_loc:end_loc].lower() == concept['concept_text'].lower():
                                concept_dict[str(j)] = {'start': start_loc,
                                                        'end': end_loc,
                                                        'text': concept['concept_text'].lower(),
                                                        'type': concept['concept_type']}
                            else:
                                print('character error')
                                print(line[:-1])
                                print(word_list)
                                print(start_loc, end_loc)
                                print(line[start_loc: end_loc])
                                print(concept)
                        else:
                            print('word error')
                            print(line[:-1])
                            print(word_list)
                            print(concept)
                            print(' '.join(word_list[word_start_loc: word_end_loc + 1]).lower())
                data_list.append({'text': line[:-1].lower(),
                                  'event_dict': concept_dict,
                                  'file_name': file_name,
                                  'line': i})
            text_file.close()
            concept_file.close()
        else:
            print(file_name)
    print('data read end!')
    return data_list


def generate_event_ner_dataset_bert(dir_name):
    max_len = 0
    tokenizer = tokenization.FullTokenizer(vocab_file='./model/uncased_L-12_H-768_A-12/vocab.txt', do_lower_case=True)
    texts = []
    labels = []
    data_list = read_data(dir_name)
    total_event = 0
    for data in data_list:
        text_list = data['text'].split('\n')
        event_dict = data['event_dict']
        all_text = list(data['text'])
        all_label = ['o' for _ in range(len(data['text']))]
        for id in event_dict:
            if event_dict[id]['type'] == '':
                continue

            total_event += 1
            if ''.join(all_text[event_dict[id]['start']:event_dict[id]['end']]) != event_dict[id]['text']:
                print(''.join(all_text[event_dict[id]['start']:event_dict[id]['end']]))
                print(event_dict[id]['text'])
            elif event_dict[id]['text'].endswith(' ') or event_dict[id]['text'].endswith('\n'):
                print(event_dict[id]['text'])

            all_label[event_dict[id]['start']] = event_dict[id]['type'] + '_b'
            for loc in range(event_dict[id]['start'] + 1, event_dict[id]['end']):
                all_label[loc] = event_dict[id]['type'] + '_i'

        for loc in range(len(all_text) - 1, -1, -1):
            if all_text[loc] in ['\n', ' ']:
                del all_text[loc]
                del all_label[loc]
        if len(all_label) != len(all_text):
            print(all_text)
            print(all_label)

        current_loc = 0
        for text in text_list:
            # print([text_list])
            # print(data['file_name'])
            # print(data['line'])
            start_loc = current_loc
            if len(text) == 0 or text is None:
                continue
            tokens = tokenizer.tokenize(text)
            if len(tokens) > max_len:
                max_len = len(tokens)
            label = []
            for token in tokens:
                label.append(all_label[current_loc])
                if token.startswith('##'):
                    if ''.join(all_text[current_loc:current_loc + len(token) - 2]).lower() != token[2:]:
                        print(''.join(all_text[current_loc:current_loc + len(token) - 2]).lower())
                        print(token[2:])
                    current_loc += (len(token) - 2)
                else:
                    if ''.join(all_text[current_loc:current_loc + len(token)]).lower() != token:
                        print(''.join(all_text[current_loc:current_loc + len(token)]).lower())
                        print(token)
                    current_loc += len(token)

            # print(''.join(all_text[start_loc:current_loc]))

            texts.append(text)
            labels.append(label)

        if current_loc != len(all_text):
            print(data['text'])

    print('total_event_num', total_event)
    # print(max_len)
    return texts, labels


if __name__ == "__main__":
    # read_data('./i2b2 2010/test_data')
    text_list, labels_list = generate_event_ner_dataset_bert('./i2b2 2010/test_data')
    print(text_list[100])
    print(labels_list[100])
    # my_file = open('./i2b2 2010/test_data/text/0427.txt')
    # print([my_file.read()])
