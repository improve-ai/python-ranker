from coolname import generate
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm
from uuid import uuid4


class SyntheticDataGenerator:

    DES_WORDS = \
        ['simple', 'free', 'love', 'embrace', 'moment', 'gratitude', 'grateful',
         'fixed', 'live', 'now', 'hard', 'together', 'kind']

    @staticmethod
    def get_synth_variants(samples_count: int) -> list:

        drawn_samples = 0
        samples = []

        while drawn_samples < samples_count:

            wc = np.random.randint(2, 5)
            curr_text = ' '.join(generate(wc))
            if not any([des_w in curr_text for des_w in SyntheticDataGenerator.DES_WORDS]):
                continue
            samples.append(
                {'text': curr_text, 'chars': len(curr_text), 'words': wc})
            drawn_samples += 1

        return samples


class SyntheticDataSampler:

    @property
    def variants(self) -> np.ndarray:
        return self._variants

    @variants.setter
    def variants(self, new_val: np.ndarray):
        self._variants = new_val

    def __init__(self, variants: list):
        self.variants = variants

    @staticmethod
    def get_norm_props(
            sample_size: int, samples_count: int, mean: int = None,
            sd: int = None):

        if not mean:
            mean = sample_size / 2

        if not sd:
            sd = sample_size / 10

        percs = np.linspace(0, sample_size, sample_size + 1)
        samples = [
            np.random.normal(mean, sd)
            for _ in range(samples_count)]

        counts, _ = (np.histogram(samples, bins=percs))

        props = [el / samples_count for el in counts]

        return props

    @staticmethod
    def get_weibull_props(
            sample_size: int, samples_count: int,
            right_sampling_limit: int, weibull_param: int):

        percs = np.linspace(0, right_sampling_limit, sample_size + 1)

        samples = np.random.weibull(weibull_param, samples_count)

        # scaling sample
        # w_min = min(samples)
        # w_max = max(samples)
        # scaled_samples = \
        #     [(s - w_min / (w_max - w_min)) * sample_size for s in samples]

        counts, _ = (np.histogram(samples, bins=percs))

        props = [el / samples_count for el in counts]

        return props

    def get_not_weighted_samples(
            self, sample_size: int, samples_count: int, props: list):

        not_weighted_variants = []

        if not props:
            p = [1/sample_size for _ in range(sample_size)]
            corr_sample_size = sample_size
        else:
            p = props
            corr_sample_size = len([el for el in p if el != 0])

        prop_dict = dict(zip([v['text'] for v in self.variants], props))

        for _ in tqdm(range(samples_count)):
            best_variant = \
                np.random.choice(
                    self.variants, replace=False, p=p)

            not_weighted_variants += \
                ([[best_variant, 1, 1]] +
                 [[v, 0, 1] for v in self.variants
                  if v['text'] != best_variant['text']])

        [nwv.append(prop_dict[nwv[0]['text']]) for nwv in not_weighted_variants]
        return not_weighted_variants, prop_dict

    def get_weighted_samples(
            self, sample_size: int, samples_count: int, props: list):

        weighted_variants = []

        if not props:
            p = [1/sample_size for _ in range(sample_size)]
            corr_sample_size = sample_size
        else:
            p = props
            corr_sample_size = len([el for el in p if el != 0])

        prop_dict = dict(zip([v['text'] for v in self.variants], props))

        for _ in tqdm(range(samples_count)):
            curr_sample = np.random.choice(self.variants, corr_sample_size, p=p)

            weighted_variants += \
                [[curr_sample[0], 1, 1],
                 [np.random.choice(curr_sample[1:]), 0, sample_size - 1]]

        [wv.append(prop_dict[wv[0]['text']]) for wv in weighted_variants]

        return weighted_variants, prop_dict

    def get_appended_samples(
            self, sample_size: int, samples_count: int, props: list):

        appended_variants = []

        if not props:
            p = [1 / sample_size for _ in range(sample_size)]
            corr_sample_size = sample_size
        else:
            p = props
            corr_sample_size = len([el for el in p if el != 0])

        prop_dict = dict(zip([v['text'] for v in self.variants], props))

        for _ in tqdm(range(samples_count)):
            curr_sample = np.random.choice(self.variants, corr_sample_size, p=p)

            appended_variants += \
                [[curr_sample[0], 1, 1]] + [[v, 0, 1] for v in self.variants]

        [wv.append(prop_dict[wv[0]['text']]) for wv in appended_variants]

        return appended_variants, prop_dict


if __name__ == '__main__':

    all_samples_sizes = [300]
    all_samples_counts = [100, 500, 1000, 2000, 4000, 5000, 10000, 20000, 30000]

    propensity_sampling_value = all_samples_counts[-1]

    right_sampling_limit = 5
    weibull_param = 10

    for sample_size in all_samples_sizes:

        norm_mean = int(sample_size / 2)
        norm_sd = int(sample_size / 10)

        norms = SyntheticDataSampler.get_norm_props(
            sample_size=sample_size, samples_count=propensity_sampling_value,
            mean=norm_mean, sd=norm_sd)

        weib = SyntheticDataSampler.get_weibull_props(
            sample_size=sample_size,
            samples_count=propensity_sampling_value,
            right_sampling_limit=right_sampling_limit,
            weibull_param=weibull_param)

        for samples_count in all_samples_counts:

            # sample_size = 300
            # samples_count = 500

            if os.path.isfile('data/synth_{}_variants.json'.format(sample_size)):
                with open('data/synth_{}_variants.json'.format(sample_size),
                          'r') as svjr:
                    synth_v_json_str = svjr.read()
                    synth_variants = json.loads(synth_v_json_str)
            else:
                synth_variants = \
                    SyntheticDataGenerator.get_synth_variants(samples_count=sample_size)

                with open('data/synth_{}_variants.json'.format(sample_size), 'w') as svj:
                    synth_v_json_str = json.dumps(synth_variants)
                    svj.write(synth_v_json_str)

            # print(synth_variants[:5].shape)
            # input('sanity check')

            sds = SyntheticDataSampler(variants=synth_variants)

            # plt.plot(range(300), norms)
            # plt.show()
            # input('pach')
            # print(norms)

            # plt.plot(range(300), weib)
            # plt.show()
            # input('pach')
            # print(norms)

            unis = [1 / sample_size for _ in range(sample_size)]

            all_props = [unis, norms, weib]
            all_props_names = \
                ['uni', 'norm_m{}_sd{}'.format(norm_mean, norm_sd),
                 'weib_rl{}_a{}'.format(right_sampling_limit, weibull_param)]

            for usd_props, usd_props_name in zip(all_props, all_props_names):

                print(usd_props_name)

                nws, nws_props = \
                    sds.get_not_weighted_samples(
                        sample_size=sample_size, samples_count=samples_count,
                        props=usd_props)

                # print(nws[:10])

                with open(
                        'data/not_weighted_data_{}_sc{}.json'.format(
                            usd_props_name, samples_count), 'w') as nwd:
                    nwd_str = json.dumps(nws)
                    nwd.write(nwd_str)

                nws_saved_props_info = {
                    'props': nws_props,
                    'case_name': usd_props_name}

                with open(
                        'data/not_weighted_props_{}_sc{}.json'.format(
                            usd_props_name, samples_count), 'w') as nwp:
                    nwp_str = json.dumps(nws_saved_props_info)
                    nwp.write(nwp_str)

                ws, ws_props = \
                    sds.get_weighted_samples(
                        sample_size=sample_size, samples_count=samples_count,
                        props=usd_props)

                # print(ws[:10])

                with open('data/weighted_data_{}_sc{}.json'.format(
                        usd_props_name, samples_count), 'w') as wd:
                    ws_str = json.dumps(ws)
                    wd.write(ws_str)

                ws_saved_props_info = {
                    'props': ws_props,
                    'case_name': usd_props_name}

                with open('data/weighted_props_{}_sc{}.json'.format(
                        usd_props_name, samples_count), 'w') as wp:
                    wp_str = json.dumps(ws_saved_props_info)
                    wp.write(wp_str)

                _as, as_props = \
                    sds.get_appended_samples(
                        sample_size=sample_size, samples_count=samples_count,
                        props=usd_props)

                with open('data/appended_data_{}_sc{}.json'.format(
                        usd_props_name, samples_count), 'w') as ad:
                    as_str = json.dumps(_as)
                    ad.write(as_str)

                as_saved_props_info = {
                    'props': as_props,
                    'case_name': usd_props_name}

                with open('data/appended_props_{}_sc{}.json'.format(
                        usd_props_name, samples_count), 'w') as ap:
                    ap_str = json.dumps(as_saved_props_info)
                    ap.write(ap_str)
