import json
import numpy as np
from onnxmltools.convert.common.data_types import FloatType, Int64TensorType, \
    FloatTensorType
from onnxmltools.convert.xgboost.convert import convert
from onnxmltools import convert_xgboost
from onnxmltools.utils import save_model
import pandas as pd
from xgboost import Booster, XGBRegressor


if __name__ == '__main__':

    # b = Booster()
    src_model_pth = '../artifacts/test_artifacts/model.xgb'
    # b.load_model(src_model_pth)
    # metadata_pth = '../test_artifacts/model.json'
    # with open(metadata_pth, 'r') as mj:
    #     json_str = mj.readline()
    #     model_metadata = json.loads(json_str)
    #
    # columns_count = len(model_metadata['table'][1])
    # feature_names = list(
    #     map(lambda i: 'f{}'.format(i), range(0, columns_count)))
    #
    # onnx_types = [(fn, FloatType) for fn in feature_names]

    # xgbm = XGBRegressor()
    # xgbm.load_model(src_model_pth)
    #
    # X = np.array([[1, 2, 3], [3, 2, 1], [1, 1, 1]])
    # y = np.array([1, 2, 3])
    #
    # xgbm.fit(X, y)
    # print(xgbm.get_booster().feature_names)
    #
    # init_types = [(fn, FloatType) for fn in xgbm.get_booster().feature_names]
    #
    # print(init_types)
    #
    # convert_xgboost(
    #     xgbm, 'sample_conv',
    #     initial_types=[('A', FloatTensorType())])

    # save_model(onnx_model, '../test_artifacts/xgb_test.onnx')
    #
    import pandas as pd
    import xgboost as xgb
    import numpy as np
    import skl2onnx.common.data_types as data_types
    import onnxmltools

    column_1 = np.array([10, 20, 30, 40, 50], dtype=int)
    column_2 = np.array([1.00, 1.20, 1.14, 9.10, 9.38], dtype=float)
    column_3 = np.array([1., 2., 3., 4., 5.], dtype=float)

    all_columns = ['0', '1', '2']

    random_dataframe = pd.DataFrame(columns=all_columns)
    random_dataframe['0'] = column_1
    random_dataframe['1'] = column_2
    random_dataframe['2'] = column_3
    print(random_dataframe)

    validate_index = 2
    x_train = random_dataframe.drop(columns=['2'])
    y_train = random_dataframe['2']
    xgb_reg = XGBRegressor(
        # objective='reg:squarederror',
        # validate_parameters=True, colsample_bytree=0.3,
        # learning_rate=0.1, max_depth=5, alpha=10, n_estimators=10
    )

    # print(xgb_reg.validate_parameters)
    xgb_reg = xgb_reg.fit(x_train, y_train)

    # input_row = np.array([[int(10), float(0.76)]])
    # print(input_row)
    #
    # prediction_result = xgb_reg.predict(input_row, validate_features=False)
    # print(prediction_result)

    xgb_reg.save_model('../test_artifacts/multiple-inputs.json')
    model_dimensions = [('A', FloatType())]
    onnx_model = convert_xgboost(xgb_reg, "multi-model", model_dimensions)
    save_model(onnx_model, '../test_artifacts/xgbreg.onnx')


