import json


if __name__ == '__main__':
    old_date_part = '10_11_2020'
    new_date_part = '16_11_2020'

    old_res_pth = \
        '../artifacts/results/{}_improve_messages_2_0_verse_sorting_results_nl.json'.format(old_date_part)
    new_res_pth = \
        '../artifacts/results/{}_improve_messages_2_0_verse_sorting_results_nl.json'.format(
        new_date_part)

    with open(old_res_pth, 'r') as orf:
        old_res_str = ''.join(orf.readlines())
        old_res = json.loads(old_res_str)

    with open(new_res_pth, 'r') as nrf:
        new_res_str = ''.join(nrf.readlines())
        new_res = json.loads(new_res_str)

    i = 0

    for old_v, new_v in zip(old_res[:20], new_res[:20]):
        print('#{} OLD | {}'.format(i, old_v['versions']['WEBM']['text']))
        print('#{} NEW | {}'.format(i, new_v['versions']['WEBM']['text']))
        print('\n')
        i += 1
