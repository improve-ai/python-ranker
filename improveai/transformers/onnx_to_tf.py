import onnx
from onnx_tf.backend import prepare

onnx_model = onnx.load("../test_artifacts/xgbreg.onnx")
tf_rep = prepare(onnx_model)
tf_rep.export_graph("../test_artifacts/output/model.pb")

# import onnx2keras
# from onnx2keras import onnx_to_keras
# import keras
# import onnx
#
# onnx_model = onnx.load('../test_artifacts/xgbreg.onnx')
# print(onnx_model)
# k_model = onnx_to_keras(onnx_model, ['A'])
