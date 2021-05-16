import coremltools as ct
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from time import time


if __name__ == '__main__':

    exp_model_pth = '../artifacts/models/12_11_2020_verses_conv.mlmodel'
    m = ct.models.MLModel(exp_model_pth)
    m_spec = m.get_spec()

    ks = ['f{}'.format(el) for el in range(len(m_spec.description.input))]
    vs = np.empty(shape=(len(ks),), dtype=float)
    vs[:] = 0.0  #  np.nan

    des_dict = dict(zip(ks, vs))

    dv = DictVectorizer(sort=True, sparse=True)
    dv.fit([des_dict])

    dv_conv = \
        ct.converters.sklearn.convert(
            dv, input_features="features",
            output_feature_names="output",)
    total_compute_time = 0
    for _ in range(100):
        st = time()
        res = dv_conv.predict({"features": {'f0': 10.0}})['output']
        res[res == 0] = np.nan
        in_dict = dict(zip(ks, res))
        pred = m.predict(data=in_dict)
        et = time()
        total_compute_time += et - st

    print(total_compute_time / 100)

