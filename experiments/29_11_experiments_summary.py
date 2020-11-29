import coremltools as ct
import numpy as np
from sklearn import datasets
from xgboost import Booster, XGBClassifier, XGBRegressor


if __name__ == '__main__':

    iris = datasets.load_iris()

    X_raw = iris.data[iris.target != 2, :3].astype(float)  # we only take the first two features.
    y_raw = iris.target[iris.target != 2].astype(int)

    # take first 5 observations from class 0 and first 2 from class 1
    # taking more than 2 from class 1 will cause python to crash
    #  Process finished with exit code 139 (interrupted by signal 11: SIGSEGV)
    X_raw_0 = X_raw[y_raw == 0, :][:5]
    X_raw_1 = X_raw[y_raw == 1, :][:2]
    y_raw_0 = y_raw[y_raw == 0][:5]
    y_raw_1 = y_raw[y_raw == 1][:2]
    # X = q
    X = np.array(
        [[float(in_el) for in_el in el] for el in X_raw_0] +
        [[float(in_el) for in_el in el] for el in X_raw_1])

    print(X[:3])
    y = [el for el in y_raw_0] + [el for el in y_raw_1]  # , 1]

    print(X)
    print(y)

    xc = XGBClassifier(missing=np.nan)
    xc.fit(X=X, y=y)

    conv1 = \
        ct.converters.xgboost.convert(
            xc, mode='classifier', feature_names={'a': [0, 1, 2]},
            class_labels=[0, 1])

    res = conv1.predict({'a': {'f0': 5.5, 'f1': 100.0}})
    print(xc.predict_proba(np.array([[5.5, np.nan, 1.0], ])))
    print(res)

    exp_mlmodel_pth = '../artifacts/models/12_11_2020_verses_conv.mlmodel'
    m = ct.models.MLModel(exp_mlmodel_pth)

    ks = ['f{}'.format(el) for el in range(len(m._spec.description.input))]
    indices = [el for el, _ in enumerate(ks)]

    exp_xgb_model_pth = '../artifacts/models/12_11_2020_verses.xgb'
    b = Booster()
    b.load_model(exp_xgb_model_pth)
    print(b.attr('user_defined_metadata'))
    b.set_attr(**{'missing': 'np.nan'})
    print(b.attr('missing'))

    # Predicting after this conversion will cause vectorization error
    conv_xgb_vect_err = \
        ct.converters.xgboost.convert(
            b, feature_names={'input_dict': list([el for el in range(len(ks))])})

    # res_1 = conv_xgb_vect_err.predict({'input_dict': {'f0': 0.0, 'f1': 1.0}})

    # Predicting after this conversion will cause:
    # Process finished with exit code 139 (interrupted by signal 11: SIGSEGV)
    conv_xgb_sig_139 = \
        ct.converters.xgboost.convert(
            b,
            feature_names={'input_dict': list([el for el in range(len(ks))])},
            mode='classifier', class_labels=[0, 1])

    # res_2 = conv_xgb_sig_139.predict({'in_feats': {'f0': 0.0, 'f1': 1.0, 'f2': 0.0}})

    conv_xgb_sig_139.user_defined_metadata['json'] = m.user_defined_metadata['json']

    conv_xgb_sig_139.save('../artifacts/models/experimental_verses.mlmodel')
