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

    raw_pth = '../data/real/bible_verses_full.json'

    with open(raw_pth, 'r') as raw_in:
        raw_str = ''.join(raw_in.readlines())
        raw_variants = json.loads(raw_str)

    raw_variants_np = \
        np.array([[raw_v, raw_v['versions']['WEBM']['text']]
                  for raw_v in raw_variants])

    srting_idxs = []
    srting_variants = []

    for srtd_var in sorted_variants[:1000]:
        for row_idx, row in enumerate(raw_variants_np):
            if row[1] == srtd_var['text']:
                srting_idxs.append(row_idx)
                srting_variants.append(srtd_var['text'])
                break

    remainder_variants = []
    for row in raw_variants_np:
        if row[1] in srting_variants:
            continue
        remainder_variants.append(row[0])

    ordered_raw = np.array(raw_variants)[srting_idxs].tolist()

    complete_list = ordered_raw + remainder_variants

    assert len(complete_list) == len(sorted_variants)

    # for srtd_true, srtd_raw in zip(sorted_variants[-10:], complete_list[-10:]):
    #     print(srtd_true)
    #     pprint(srtd_raw)

    # input('sanity check')

    trgt_pth = \
        '../results/14_10_2020_improve_messages_2_0_verse_sorted_nl.json'

    with open(trgt_pth, 'w') as trgt_json:
        saved_str = simplejson.dumps(ordered_raw, indent=4)
        trgt_json.write(saved_str)

    # # print('highest scored variant')
    # print(sorted_variants[:10])
    # # print('raw json example elemnt')
    # pprint(ordered_raw[:10])

    pass
