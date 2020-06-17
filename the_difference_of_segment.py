import i2b2_2010_data_preprocess
import i2b2_2012_data_preprocess

event_list_2010 = {}
data_list = i2b2_2010_data_preprocess.read_data('./i2b2_2010/train_data')
for data in data_list:
    event_dict = data['event_dict']
    for m_id in event_dict:
        if event_list_2010.get(event_dict[m_id]['type'].lower()) is None:
            event_list_2010[event_dict[m_id]['type'].lower()] = []
        event_list_2010[event_dict[m_id]['type'].lower()].append(event_dict[m_id]['text'])
data_list = i2b2_2010_data_preprocess.read_data('./i2b2_2010/test_data')
for data in data_list:
    event_dict = data['event_dict']
    for m_id in event_dict:
        if event_list_2010.get(event_dict[m_id]['type'].lower()) is None:
            event_list_2010[event_dict[m_id]['type'].lower()] = []
        event_list_2010[event_dict[m_id]['type'].lower()].append(event_dict[m_id]['text'])

print(len(event_list_2010))
for m_type in event_list_2010:
    print(m_type, len(event_list_2010[m_type]))
    # print(event_list_2010[m_type])


event_list_2012 = {}
data_list = i2b2_2012_data_preprocess.read_data_dir('./i2b2_2012/train_data')
for data in data_list:
    event_dict = data['event_dict']
    for m_id in event_dict:
        if event_list_2012.get(event_dict[m_id]['type'].lower()) is None:
            event_list_2012[event_dict[m_id]['type'].lower()] = []
        event_list_2012[event_dict[m_id]['type'].lower()].append(event_dict[m_id]['text'].lower())
data_list = i2b2_2012_data_preprocess.read_data_dir('./i2b2_2012/test_data')
for data in data_list:
    event_dict = data['event_dict']
    for m_id in event_dict:
        if event_list_2012.get(event_dict[m_id]['type'].lower()) is None:
            event_list_2012[event_dict[m_id]['type'].lower()] = []
        event_list_2012[event_dict[m_id]['type'].lower()].append(event_dict[m_id]['text'].lower())

print(len(event_list_2012))
for m_type in event_list_2012:
    print(m_type, len(event_list_2012[m_type]))
    # print(event_list_2012[m_type])


# 计算A和B两个概念间的重叠字符长度
def count_overlap_length(a, b):
    if a.find(b) != -1:
        return len(b)
    if b.find(a) != -1:
        return len(a)
    for i in range(1, len(a)):
        if a[i - 1] == ' ' and str(b).startswith(a[i:]):
            return len(a) - i
        if a[len(a) - i] == ' ' and str(b).endswith(a[:len(a) - i]):
            return len(a) - i
    return 0


def count_distribution(event_list_a, event_list_b):
    for m_type in event_list_a:
        if event_list_b.get(m_type) is not None:
            print(m_type)
            overlap_num = 0
            all_included_num = 0
            all_include_num = 0
            included_num = 0
            include_num = 0
            for event_a in event_list_a[m_type]:
                has_overlap = False
                is_included = False
                include = False
                for event_b in event_list_b[m_type]:
                    overlap_length = count_overlap_length(event_a, event_b)
                    if overlap_length != 0:
                        has_overlap = True
                    if overlap_length == len(event_a):
                        is_included = True
                    if overlap_length == len(event_b):
                        include = True
                if has_overlap:
                    overlap_num += 1
                if is_included and not include:
                    all_included_num += 1
                if not is_included and include:
                    all_include_num += 1
                if is_included:
                    included_num += 1
                if include:
                    include_num += 1
            print('overlap_num', overlap_num)
            print('included_num', included_num)
            print('all_included_num', all_included_num)
            print('include_num', include_num)
            print('all_include_num', all_include_num)
