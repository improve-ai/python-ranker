from coremltools.proto import Model_pb2 as _Model_pb2
from coremltools.models import datatypes
from coremltools.models import MLModel as _MLModel
from coremltools.models._interface_management import \
    set_transform_interface_params
from coremltools import SPECIFICATION_VERSION
from coremltools.converters.sklearn import convert
from coremltools.models.feature_vectorizer import create_feature_vectorizer

import numpy as np


if __name__ == '__main__':

    feat_count = 10

    feat_v_inputs = [('f{}'.format(el), datatypes.Double) for el in range(3)]

    # An example of feature vectorizer which raises and error when features are
    # missing
    spec, _output_dimension = \
        create_feature_vectorizer(feat_v_inputs, 'out')

    model_v = _MLModel(spec)
    # All features present - no error
    print(model_v.predict({'f0': 0, 'f1': 0, 'f2': np.nan}))

    # Missing feature f2 - error
    try:
        print(model_v.predict({'f0': 0, 'f1': 0, 'f10': np.nan}))
    except Exception as exc:
        print(exc)

    # creating input features for Imputer constructed from compiled
    # protobuf
    # Imputer will expect a dict as an input and will look for an array to be
    # imputed under 'in_feats' key
    input_feats = {'in_feats': range(feat_count)}
    # Output will be a dict and will have an array with results stored under
    # 'out_feats' key
    output_feats = {'out_feats': range(feat_count)}

    raw_model_spec = _Model_pb2.Model()
    raw_model_spec.specificationVersion = \
        SPECIFICATION_VERSION

    # this sets input and output definition to the proper attributes of
    # protobuf object
    io_set_model_spec = set_transform_interface_params(
        raw_model_spec, input_features=input_feats, output_features=output_feats)  # ,

    # this is the section where Imputer's attributed are set
    # (they are corresponding to its protobuf definition)
    imputer_spec = io_set_model_spec.imputer

    # Here 'replacement' values are set
    for v in range(feat_count):
        imputer_spec.imputedDoubleArray.vector.append(v)

    # Here missing value is set -> if this value is encountered it will be
    # replaced
    imputer_spec.replaceDoubleValue = float(np.nan)

    # fianlly creatig mlmodel from constructed spec
    model = _MLModel(io_set_model_spec)

    # this call will return input dict
    print(model.predict(
        {'in_feats': {'f1': 0}}))

    # this call will return imputer array
    print(model.predict(
        {'in_feats': np.array(
            [1, 1, 1] + [np.nan for el in range(feat_count - 3)])}))
