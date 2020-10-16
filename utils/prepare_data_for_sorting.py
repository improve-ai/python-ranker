from copy import deepcopy
import json

if __name__ == '__main__':

    src_json_pth = '../data/real/bible_verses_full.json'

    with open(src_json_pth, 'r') as srcj:
        read_str = ''.join(srcj.readlines())
        all_variants = json.loads(read_str)

    fxd_variants = []
    for variant in all_variants:
        fxd_variants.append({
            'ref': variant['ref'],
            'text': variant['versions']['WEBM']['text']})

    trgt_json_pth = '../data/real/2bs_bible_verses_full.json'

    with open(trgt_json_pth, 'w') as trgtj:
        wrtn_str = json.dumps(fxd_variants, encoding='utf-8')
        trgtj.writelines(wrtn_str)
