import tensorflow as tf
import tokenization
import time


def get_tar_event(text, labels, label_list):
    label_map = {}
    for i, lab in enumerate(label_list):
        label_map[lab] = i

    current_type = None
    current_start_loc = None

    event_list = []
    for i, token in enumerate(text):
        label = labels[i]
        for j, lab in enumerate(label):
            if lab == 1 and label_list[j].endswith('_b'):
                current_type = label_list[j][:-2]
                current_start_loc = i
        for j, lab in enumerate(label):
            if lab == 1 and label_list[j].endswith('_e'):
                if current_start_loc is not None:
                    event_list.append({'event': text[current_start_loc:i + 1],
                                       'event_type': current_type,
                                       'start_loc': current_start_loc,
                                       'end_loc': i})
                else:
                    print(text, labels)
    return event_list


def get_pre_event_slide_window(text, labels, label_list):
    candidate_event_list = []
    # print(text)
    # print(labels)
    # print(len(labels))
    id = 0
    for i in range(10):
        for j in range(128 - i):
            # print(i, j, labels[id])
            for k in range(len(label_list)):
                if labels[id][k] >= 0.5:
                    candidate_event_list.append({'event': text[j: i + j + 1],
                                                 'event_type': label_list[k],
                                                 'start_loc': j,
                                                 'end_loc': i + j,
                                                 'prob': labels[id][k]})
            id += 1
    # print(id)

    best_result_record = [
        {"score": 0, 'event_start': None, 'event_end': None, 'type': None, 'previous_loc': i} for i in
        range(len(text) + 1)]
    for i, token in enumerate(text):
        best_result_record[i + 1] = best_result_record[i].copy()
        for j, event in enumerate(candidate_event_list):
            if event['end_loc'] == i and (best_result_record[event['start_loc']]['score'] + event['prob']) > best_result_record[i + 1]['score']:
                best_result_record[i + 1]['score'] = best_result_record[event['start_loc']]['score'] + event['prob']
                best_result_record[i + 1]['event_start'] = event['start_loc']
                best_result_record[i + 1]['event_end'] = event['end_loc']
                best_result_record[i + 1]['type'] = event['event_type']
                best_result_record[i + 1]['previous_loc'] = event['start_loc']

    t = len(text)
    event_list = []
    # for i in range(len(text) + 1):
    #     print(best_result_record[i])
    while best_result_record[t]['type'] is not None and best_result_record[t]['previous_loc'] != 0:
        # print(best_result_record[t])
        start_loc = best_result_record[t]['event_start']
        end_loc = best_result_record[t]['event_end']
        event_list.append({'event': text[start_loc: end_loc + 1],
                           'event_type': best_result_record[t]['type'],
                           'start_loc': start_loc,
                           'end_loc': end_loc})
        t = best_result_record[t]['previous_loc']
    return event_list


def evaluate_with_slide_window(result, label_list, tokenizer, task_name, is_write=True):
    with tf.gfile.GFile("./result/" + str(task_name) + "_slide_window_evaluate.results", "w") as writer:
        num_written_lines = 0

        tar_event_num = 0
        pre_event_num = 0
        cor_event_num = 0
        cor_tar_event_num = 0
        cor_pre_event_num = 0

        tar_event_num_per_type = {}
        pre_event_num_per_type = {}
        cor_event_num_per_type = {}
        cor_tar_event_num_per_type = {}
        cor_pre_event_num_per_type = {}

        for (i, prediction) in enumerate(result):
            pre_label_ids = prediction["pre_label_prob"].tolist()
            label_ids = prediction["label_ids"].tolist()
            input_ids = prediction["input_ids"].tolist()
            tar_label_ids = prediction["tar_label_ids"].tolist()
            input_text = tokenizer.convert_ids_to_tokens(input_ids)

            tar_event_list = get_tar_event(input_text, tar_label_ids,
                                           ['EVIDENTIAL_b', 'EVIDENTIAL_e', 'OCCURRENCE_b', 'OCCURRENCE_e', 'PROBLEM_b',
                                            'PROBLEM_e', 'TEST_b', 'TEST_e', 'TREATMENT_b', 'TREATMENT_e',
                                            'CLINICAL_DEPT_b', 'CLINICAL_DEPT_e', '[CLS]', '[SEP]'])
            pre_event_list = get_pre_event_slide_window(input_text, pre_label_ids, label_list)

            current_tar_event_num = len(tar_event_list)
            current_pre_event_num = len(pre_event_list)
            current_cor_event_num = 0
            current_cor_tar_event_num = 0
            current_cor_pre_event_num = 0

            tar_event_num += len(tar_event_list)
            pre_event_num += len(pre_event_list)

            for tar_event in tar_event_list:
                word_type = tar_event['event_type']
                tar_event_num_per_type[word_type] = tar_event_num_per_type.get(word_type, 0) + 1

            for pre_event in pre_event_list:
                word_type = pre_event['event_type']
                pre_event_num_per_type[word_type] = pre_event_num_per_type.get(word_type, 0) + 1

            for tar_event in tar_event_list:
                for pre_event in pre_event_list:
                    if tar_event['start_loc'] == pre_event['start_loc'] and \
                            tar_event['end_loc'] == pre_event['end_loc'] and \
                            tar_event['event_type'] == pre_event['event_type']:
                        word_type = tar_event['event_type']
                        cor_event_num += 1
                        current_cor_event_num += 1
                        cor_event_num_per_type[word_type] = cor_event_num_per_type.get(word_type, 0) + 1
                        break

            for tar_event in tar_event_list:
                for pre_event in pre_event_list:
                    if min(tar_event['end_loc'], pre_event['end_loc']) >= max(tar_event['start_loc'],
                                                                              pre_event['start_loc']) and \
                            tar_event['event_type'] == pre_event['event_type']:
                        word_type = tar_event['event_type']
                        cor_tar_event_num += 1
                        current_cor_tar_event_num += 1
                        cor_tar_event_num_per_type[word_type] = cor_tar_event_num_per_type.get(word_type, 0) + 1
                        break

            for pre_event in pre_event_list:
                for tar_event in tar_event_list:
                    if min(tar_event['end_loc'], pre_event['end_loc']) >= max(tar_event['start_loc'],
                                                                              pre_event['start_loc']) and \
                            tar_event['event_type'] == pre_event['event_type']:
                        word_type = tar_event['event_type']
                        cor_pre_event_num += 1
                        current_cor_pre_event_num += 1
                        cor_pre_event_num_per_type[word_type] = cor_tar_event_num_per_type.get(word_type, 0) + 1
                        break
            if current_cor_event_num != current_tar_event_num:
                output_line = 'sentence_num:' + str(num_written_lines) + '\n' + \
                              "target_word_num:" + str(current_tar_event_num) + \
                              "\tpredicate_word_num:" + str(current_pre_event_num) + \
                              "correct_word_num:" + str(current_cor_event_num) + "\n" + \
                              "\tcorrect_tar_word_num:" + str(current_cor_tar_event_num) + \
                              "\tcorrect_pre_word_num:" + str(current_cor_pre_event_num) + "\n" + \
                              "".join(["{:<50}".format(str(class_probability).lower()) for class_probability in
                                       input_text]) + "\n" + str(tar_event_list) + '\n' + str(pre_event_list) + '\n'
                writer.write(output_line)
                num_written_lines += 1

    accuracy = cor_event_num / pre_event_num
    recall = cor_event_num / tar_event_num
    if is_write:
        with open("./result/" + str(task_name) + "_slide_window_strict_evaluate.record", "a") as file:
            print("=============================strict standard============================", file=file)
            print("=============================", time.ctime(), "============================", file=file)
            print("accuracy\t" + str(accuracy), file=file)
            print("recall\t" + str(recall), file=file)
            print("f1\t" + str(2 * accuracy * recall / (accuracy + recall)), file=file)
            print('total_word_num', tar_event_num, file=file)
            print('pre_word_num', pre_event_num, file=file)
            print('correct_word_num', cor_event_num, file=file)
            for word_type in tar_event_num_per_type:
                correct_word_num = cor_event_num_per_type.get(word_type, 0)
                total_word_num = tar_event_num_per_type.get(word_type, 0)
                pred_word_num = pre_event_num_per_type.get(word_type, 0)
                accuracy = correct_word_num / pred_word_num
                recall = correct_word_num / total_word_num
                print('----------------' + word_type + '-------------------', file=file)
                print("accuracy\t" + str(accuracy), file=file)
                print("recall\t" + str(recall), file=file)
                print("f1\t" + str(2 * accuracy * recall / (accuracy + recall)), file=file)
                print('total_word_num', total_word_num, file=file)
                print('pre_word_num', pred_word_num, file=file)
                print('correct_word_num', correct_word_num, file=file)

        accuracy = cor_pre_event_num / pre_event_num
        recall = cor_tar_event_num / tar_event_num
        with open("./result/" + str(task_name) + "_slide_window_relax_evaluate.record", "a") as file:
            print("=============================relax standard============================", file=file)
            print("=============================", time.ctime(), "============================", file=file)
            print("accuracy\t" + str(accuracy), file=file)
            print("recall\t" + str(recall), file=file)
            print("f1\t" + str(2 * accuracy * recall / (accuracy + recall)), file=file)
            print('pre_correct_word', cor_pre_event_num, file=file)
            print('pre_total_word', pre_event_num, file=file)
            print('tar_correct_word', cor_tar_event_num, file=file)
            print('tar_total_word', tar_event_num, file=file)
            for word_type in tar_event_num_per_type:
                tar_correct_word_num = cor_tar_event_num_per_type.get(word_type, 0)
                tar_total_word_num = tar_event_num_per_type.get(word_type, 0)
                pre_correct_word_num = cor_pre_event_num_per_type.get(word_type, 0)
                pre_total_word_num = pre_event_num_per_type.get(word_type, 0)

                accuracy = pre_correct_word_num / pre_total_word_num
                recall = tar_correct_word_num / tar_total_word_num
                print('----------------' + word_type + '-------------------', file=file)
                print("accuracy\t" + str(accuracy), file=file)
                print("recall\t" + str(recall), file=file)
                print("f1\t" + str(2 * accuracy * recall / (accuracy + recall)), file=file)
                print('tar_correct_word_num', tar_correct_word_num, file=file)
                print('tar_total_word_num', tar_total_word_num, file=file)
                print('pre_correct_word_num', pre_correct_word_num, file=file)
                print('pre_total_word_num', pre_total_word_num, file=file)
