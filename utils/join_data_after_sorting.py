import json
import numpy as np
from pprint import pprint
import simplejson


if __name__ == '__main__':

    sorted_pth = \
        '../results/14_10_2020_improve_messages_2_0_verse_sorting_results.json'

    with open(sorted_pth, 'r') as json_in:
        sorted_str = ''.join(json_in.readlines())
        sorted_variants = json.loads(sorted_str)

    # print(sorted_variants[:10])
    # input('a')

    raw_pth = '../data/real/bible_verses_full.json'

    with open(raw_pth, 'r') as raw_in:
        raw_str = ''.join(raw_in.readlines())
        raw_variants = json.loads(raw_str)

    raw_variants_np = \
        np.array([[raw_v, raw_v['versions']['WEBM']['text'], raw_v['ref']]
                  for raw_v in raw_variants])

    srting_idxs = []
    srting_variants = []
    srting_refs = []
    duplicates = []

    for srtd_var in sorted_variants[:1000]:
        for row_idx, row in enumerate(raw_variants_np):

            if row[1] == srtd_var['text'] and row[2] == srtd_var['ref'] \
                    and row_idx in srting_idxs:
                print('{} + {}'.format(row[1], row[2]))

            if row[1] == srtd_var['text'] and row[2] == srtd_var['ref'] \
                    and row_idx not in srting_idxs:
                srting_idxs.append(row_idx)
                srting_variants.append(srtd_var['text'])
                srting_refs.append(srtd_var['ref'])
                break

    # print(len(srting_variants))
    # print(len(np.unique(srting_variants)))
    # print(len(np.unique([el['versions']['WEBM']['text'] + el['ref'] for el in raw_variants])))
    # print(len(raw_variants))
    # input('abc')

    remainder_variants_idxs = \
        [row_idx for row_idx, _ in enumerate(raw_variants_np)
         if row_idx not in srting_idxs]

    remainder_variants = \
        (raw_variants_np[remainder_variants_idxs][:, 0]).tolist()
    # print(remainder_variants[:10])
    # input('sanit')

    # for row in raw_variants_np:
    #     if row[1] in srting_variants:
    #         continue
    #     remainder_variants.append(row[0])

    ordered_raw = np.array(raw_variants)[srting_idxs].tolist()

    complete_list = ordered_raw + remainder_variants

    assert len(complete_list) == len(raw_variants)

    # for srtd_true, srtd_raw in zip(sorted_variants[-10:], complete_list[-10:]):
    #     print(srtd_true)
    #     pprint(srtd_raw)

    # input('sanity check')

    q = [el['versions']['WEBM']['text'] + el['ref'] for el in complete_list]
    qq = [el['versions']['WEBM']['text'] + el['ref'] for el in raw_variants]

    print(len(set(q)))
    print(len(set(qq)))

    print(len(complete_list))
    print(len(raw_variants))

    input('sanity check')

    # pprint(sorted_variants[3])
    # pprint(complete_list[1001])
    # pprint(remainder_variants[1])
    # input('sanity check')

    trgt_pth = \
        '../results/14_10_2020_improve_messages_2_0_verse_sorted_nl_01.json'

    with open(trgt_pth, 'w') as trgt_json:
        saved_str = simplejson.dumps(complete_list, indent=4)
        trgt_json.write(saved_str)

    # # print('highest scored variant')
    # print(sorted_variants[:10])
    # # print('raw json example elemnt')
    # pprint(ordered_raw[:10])

    pass



# import json
# import numpy as np
# from pprint import pprint
# import simplejson
#
#
# if __name__ == '__main__':
#
#     sorted_pth = \
#         '../results/14_10_2020_improve_messages_2_0_verse_sorted_nl.json'
#
#     with open(sorted_pth, 'r') as json_in:
#         sorted_str = ''.join(json_in.readlines())
#         sorted_variants = json.loads(sorted_str)
#
#     raw_pth = '../data/real/bible_verses_full.json'
#
#     with open(raw_pth, 'r') as raw_in:
#         raw_str = ''.join(raw_in.readlines())
#         raw_variants = json.loads(raw_str)
#
#     raw_variants_np = \
#         np.array([[raw_v, raw_v['versions']['WEBM']['text'], raw_v['ref']]
#                   for raw_v in raw_variants])
#
#     srting_idxs = []
#     srting_variants = []
#
#     duplicates = []
#
#     for srtd_var in sorted_variants:
#         for row_idx, row in enumerate(raw_variants_np):
#             # if row[1] == srtd_var['versions']['WEBM']['text'] \
#             #         and row_idx in srting_idxs:
#             #     duplicates.append(row_idx)
#
#             if row[1] == srtd_var['versions']['WEBM']['text'] \
#                     and row_idx not in srting_idxs:
#                 srting_idxs.append(row_idx)
#                 srting_variants.append(srtd_var['versions']['WEBM']['text'])
#                 break
#
#     # print('len(set(srting_idxs))')
#     # print(len(set(srting_idxs)))
#     # # print(len(sorted_variants))
#     # qqq = [el['versions']['WEBM']['text'] for el in raw_variants]
#     # print(len(np.unique(qqq)))
#
#     # print(duplicates)
#
#     # for el_idx, el in enumerate(raw_variants_np):
#     #     if el[1] == 'Grace to you and peace from God our Father and the Lord Jesus Christ.':
#     #         print(el_idx)
#
#     # pprint(raw_variants_np[532][0])
#     # pprint(raw_variants_np[815][0])
#     # pprint(sorted_variants[0])
#     # pprint(sorted_variants[3])
#     # pprint(sorted_variants[19])
#     # pprint(sorted_variants[999])
#
#     remainder_variants = []
#     for row in raw_variants_np:
#         if row[1] in srting_variants:
#             continue
#         remainder_variants.append(row[0])
#
#     ordered_raw = np.array(raw_variants)[srting_idxs].tolist()
#
#     complete_list = ordered_raw + remainder_variants
#     # complete_list = srting_variants + remainder_variants
#
#     # assert len(complete_list) == len(sorted_variants)
#
#     # print(len(complete_list))
#     # print(len(raw_variants))
#     #
#     # q = [el['versions']['WEBM']['text'] for el in complete_list]
#     # qq = [el['versions']['WEBM']['text'] for el in raw_variants]
#     # w = set(q)
#     # ww = set(qq)
#     # print(len(w))
#     # print(len(ww))
#
#     assert len(complete_list) == len(raw_variants)
#
#     # for srtd_true, srtd_raw in zip(sorted_variants[-10:], complete_list[-10:]):
#     #     print(srtd_true)
#     #     pprint(srtd_raw)
#
#     # input('sanity check')
#
#     trgt_pth = \
#         '../results/14_10_2020_improve_messages_2_0_verse_sorted_nl_01.json'
#
#     with open(trgt_pth, 'w') as trgt_json:
#         saved_str = simplejson.dumps(complete_list, indent=4)
#         trgt_json.write(saved_str)
#
#     # # print('highest scored variant')
#     # print(sorted_variants[:10])
#     # # print('raw json example elemnt')
#     # pprint(ordered_raw[:10])
#
#     pass
