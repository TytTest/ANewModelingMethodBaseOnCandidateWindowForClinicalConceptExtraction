import os
import re
import slide_window_model_evaluate
import tokenization
import tensorflow


def read_xml_file(file_name):
    file = open(file_name, 'r', encoding='utf-8')
    contents = file.read()

    events = re.findall('<EVENT[^>]*/>', contents)
    event_dict = {}
    for event in events:
        id = re.search('id="([^"]*)"', event).group(1)
        start = re.search('start="([^"]*)"', event).group(1)
        end = re.search('end="([^"]*)"', event).group(1)
        text = re.search('text="([^"]*)"', event).group(1)
        modality = re.search('modality="([^"]*)"', event).group(1)
        polarity = re.search('polarity="([^"]*)"', event).group(1)
        type = re.search('type="([^"]*)"', event).group(1)
        event_dict[id] = {'start': int(start),
                          'end': int(end),
                          'text': text,
                          'modality': modality,
                          'polarity': polarity,
                          'type': type}

    timex3s = re.findall('<TIMEX3[^>]*/>', contents)
    timex3_dict = {}
    for timex3 in timex3s:
        id = re.search('id="([^"]*)"', timex3).group(1)
        start = re.search('start="([^"]*)"', timex3).group(1)
        end = re.search('end="([^"]*)"', timex3).group(1)
        text = re.search('text="([^"]*)"', timex3).group(1)
        type = re.search('type="([^"]*)"', timex3).group(1)
        val = re.search('val="([^"]*)"', timex3).group(1)
        mod = re.search('mod="([^"]*)"', timex3).group(1)
        timex3_dict[id] = {'start': int(start),
                           'end': int(end),
                           'text': text,
                           'type': type,
                           'val': val,
                           'mod': mod}

    tlinks = re.findall('<TLINK[^>]*/>', contents)
    tlink_list = []
    for tlink in tlinks:
        id = re.search('id="([^"]*)"', tlink).group(1)
        fromID = re.search('fromID="([^"]*)"', tlink).group(1)
        fromText = re.search('fromText="([^"]*)"', tlink).group(1)
        toID = re.search('toID="([^"]*)"', tlink).group(1)
        toText = re.search('toText="([^"]*)"', tlink).group(1)
        type = re.search('type="([^"]*)"', tlink).group(1)
        tlink_list.append({'id': id,
                           'fromID': fromID,
                           'fromText': fromText,
                           'toID': toID,
                           'toText': toText,
                           'type': type})

    sectimes = re.findall('<SECTIME[^>]*/>', contents)
    sectime_dict = {}
    for sectime in sectimes:
        id = re.search('id="([^"]*)"', sectime).group(1)
        start = re.search('start="([^"]*)"', sectime).group(1)
        end = re.search('end="([^"]*)"', sectime).group(1)
        text = re.search('text="([^"]*)"', sectime).group(1)
        type = re.search('type="([^"]*)"', sectime).group(1)
        dvalue = re.search('dvalue="([^"]*)"', sectime).group(1)
        sectime_dict[id] = {'start': int(start),
                            'end': int(end),
                            'text': text,
                            'type': type,
                            'dvalue': dvalue}

    text = re.search('<TEXT>([\s\S]*)</TEXT>', contents).group(1)[9:-4]

    return text, event_dict, timex3_dict, sectime_dict, tlink_list


def read_data_dir(dir_name):
    dir_list = os.listdir(dir_name)
    data_list = []
    for cur_file in dir_list:
        # 获取文件的绝对路径
        path = os.path.join(dir_name, cur_file)
        if os.path.isfile(path) and path.endswith('.xml'):  # 判断是否是文件还是目录需要用绝对路径
            text, event_dict, timex3_dict, sectime_dict, tlink_list = read_xml_file(path)
            data_list.append({'text': text,
                              'event_dict': event_dict,
                              'timex3_dict': timex3_dict,
                              'sectime_dict': sectime_dict,
                              'tlink_list': tlink_list,
                              'file': path})
    return data_list


def generate_sequence_labeling_dataset(dir_name, tokenizer=None):
    if tokenizer is None:
        tokenizer =
    sentence_list = []
    labels_list = []

    data_list = read_data_dir(dir_name)
    for data in data_list:
        event_dict = data['event_dict']
        for event_id in event_dict:



# 生成滑动窗口模型的数据集， 由于滑动窗口模型会排除长度大于最大窗口长度的事件，因此还需要使用be_tag来记录完整的事件列表
def generate_slide_window_dataset(dir_name, be_tag_list, sw_tag_list, max_window_length, max_sequence_length, tokenizer):
    be_tag_map = {}
    for i, lab in enumerate(be_tag_list):
        be_tag_map[lab] = i
    sw_tag_map = {}
    for i, lab in enumerate(sw_tag_list):
        sw_tag_map[lab] = i
    texts = []
    sw_labels = []
    be_labels = []

    data_list = read_data_dir(dir_name)
    total_event = 0

    for data in data_list:
        event_dict = data['event_dict']

        character_list = list(data['text'])
        character_labels = ['o' for _ in range(len(data['text']))]

        # 生成基于字的标签
        for id in event_dict:
            if event_dict[id]['type'] == '':
                print('without event type!', event_dict[id])
                continue

            total_event += 1

            if ''.join(character_list[event_dict[id]['start']:event_dict[id]['end']]) != event_dict[id]['text']:
                print('The location of the event is incorrect.')
                print(''.join(character_list[event_dict[id]['start']:event_dict[id]['end']]))
                print(event_dict[id]['text'])
            elif event_dict[id]['text'].endswith(' ') or event_dict[id]['text'].endswith('\n'):
                print('The event contains illegal characters.', event_dict[id]['text'])

            if event_dict[id]['start'] == event_dict[id]['end'] - 1:
                character_labels[event_dict[id]['start']] = event_dict[id]['type'] + '_be'
            else:
                character_labels[event_dict[id]['start']] = event_dict[id]['type'] + '_b'
                character_labels[event_dict[id]['end'] - 1] = event_dict[id]['type'] + '_e'

        # 去除文本中的换行符与空格，便于文本与tokeniz后的文本对齐
        for loc in range(len(character_list) - 1, -1, -1):
            if character_list[loc] in ['\n', ' ']:
                del character_list[loc]
                del character_labels[loc]
        if len(character_labels) != len(character_list):
            print(character_list)
            print(character_labels)

        # print(data['file'])
        # for i, text in enumerate(all_text):
        #     print("{:<5}".format(text), all_label[i])

        current_loc = 0
        text_list = data['text'].split('\n')

        # 生成每个句子tokeniz后的be tag标签
        for text in text_list:
            if len(text) == 0 or text is None:
                continue
            tokens = tokenizer.tokenize(text)
            be_label = []
            for token in tokens:
                temp_loc = current_loc
                if token.startswith('##'):
                    if ''.join(character_list[current_loc:current_loc + len(token) - 2]).lower() != token[2:]:
                        print(''.join(character_list[current_loc:current_loc + len(token) - 2]).lower())
                        print(token[2:])
                    current_loc += (len(token) - 2)
                else:
                    if ''.join(character_list[current_loc:current_loc + len(token)]).lower() != token:
                        print(''.join(character_list[current_loc:current_loc + len(token)]).lower())
                        print(token)
                    current_loc += len(token)

                temp_label = [0 for _ in range(len(be_tag_list))]
                for k in range(temp_loc + 1, current_loc - 1):
                    if character_labels[k].endswith('_b') or character_labels[k].endswith('_e'):
                        print('The token contains an error label.')
                        print(tokens)
                        print(character_list)
                        print(character_labels)
                        print(token)
                        break
                if character_labels[temp_loc].endswith('_be'):
                    temp_label[be_tag_map[character_labels[temp_loc][:-2] + 'b']] = 1
                    temp_label[be_tag_map[character_labels[current_loc - 1][:-2] + 'e']] = 1
                if character_labels[temp_loc].endswith('_b'):
                    temp_label[be_tag_map[character_labels[temp_loc]]] = 1
                if character_labels[current_loc - 1].endswith('_e'):
                    temp_label[be_tag_map[character_labels[current_loc - 1]]] = 1
                be_label.append(temp_label)
            be_labels.append(be_label)
            # print(''.join(all_text[start_loc:current_loc]))
            # for i, token in enumerate(tokens):
            #     print("{:<15}".format(token), label[i])

            # 通过be tag从句子中取出事件列表，这个事件列表包括每个事件在tokens中的位置，用于产生sw tag
            event_list = slide_window_model_evaluate.get_tar_event(tokens, be_label, be_tag_list)

            # 初始化sw tag
            label_matrix = [[[0 for _ in range(len(sw_tag_list))] for _ in range(max_sequence_length)] for _ in
                            range(max_window_length)]
            # print(event_list)

            for event in event_list:
                # print(event)
                if event['end_loc'] > max_sequence_length - 2:
                    # print('End of event exceeds maximum length of sequence.', event)
                    continue
                event_length = event['end_loc'] - event['start_loc']
                start_loc = event['start_loc'] + 1
                if event_length >= max_window_length:
                    # print('The length of the event exceeds the maximum length of the window.', event)
                    continue
                label_matrix[event_length][start_loc][sw_tag_map[event['event_type']]] = 1
            be_label = []
            for k in range(max_window_length):
                if k != 0:
                    be_label.extend(label_matrix[k][:-k])
                    # print(len(label_matrix[k][:-k]))
                else:
                    be_label.extend(label_matrix[k])
                    # print(len(label_matrix[k]))
            # print(len(label))
            # tar_event_list = bert_evaluate.get_tar_event_slide_window(tokens, label, label_list)
            # print(tar_event_list)
            texts.append(text)
            sw_labels.append(be_label)
        if current_loc != len(character_list):
            print(data['text'])

    print('total_event_num', total_event)
    # print(max_len)
    return texts, sw_labels, be_labels
