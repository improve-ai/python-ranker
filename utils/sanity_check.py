import json
from pprint import pprint


if __name__ == '__main__':
    original_data_pth = '../data/real/bible_verses_full.json'

    with open(original_data_pth, 'r') as oj:
        orig_str = ''.join(oj.readlines())
        orig = json.loads(orig_str)

    results_data_pth = \
        '../results/14_10_2020_improve_messages_2_0_verse_sorted_nl_01.json'

    with open(results_data_pth, 'r') as rj:
        res_str = ''.join(rj.readlines())
        res = json.loads(res_str)

    print(len(orig))
    print(len(res))

    for el in res[:20]:
        pprint(el)
