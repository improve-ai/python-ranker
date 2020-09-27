# Requires Python3.7, coremltools doens't support Python3.8
import os
import coremltools
import json
import tarfile
import uuid
from io import BytesIO
import shutil
import pickle as pkl

# Input:
# `xgboost_tar_gz_buffer` is a data buffer for a tar.gz file
# that contains a file named 'xgboost-model'
# `metadata` is the decoded metadata JSON
#
# Output: A buffer of .tar.gz object containing
# xgboost model, coreml model, and metadata json.
def transform_model(xgboost_tar_gz_buffer, metadata):
  # filter only the final keys
  metadata = dict((k, metadata[k]) for k in ['table', 'namespaces', 'hashed_feature_count', 'model_seed'] if k in metadata)
  
  tmp_folder = f'/tmp/{uuid.uuid4()}/'
  os.makedirs(tmp_folder, exist_ok=True)

  output_path = tmp_folder + "out.tar.gz"
  xboots_fileobj = BytesIO(xgboost_tar_gz_buffer)

  try:
    with tarfile.open(fileobj=xboots_fileobj, mode="r:gz") as input_tar, \
         tarfile.open(name=output_path, mode="x:gz") as output_tar:

      # Pack metadata json
      json_bytes = json.dumps(metadata).encode('utf8')
      json_info = tarfile.TarInfo("model.json")
      json_info.size = len(json_bytes)
      output_tar.addfile(json_info, fileobj=BytesIO(json_bytes))

      for member in input_tar.getmembers():
        if not member.isfile() or member.name != "xgboost-model":
          continue

        # Pack xgb model
        xgb_contents = input_tar.extractfile(member)
        bytes = bytearray(xgb_contents.read(member.size))
        # Use pickle to load models provided by AWS SageMaker
        booster = pkl.loads(bytes)
        xgb_path = tmp_folder + "model.xgb"
        # This will save model without pickle
        booster.save_model(xgb_path)
        output_tar.add(xgb_path, arcname="model.xgb")

        # Pack coreml model
        columns_count = len(metadata['table'][1])
        feature_names = list(map(lambda i: 'f{}'.format(i), range(0, columns_count)))

        cml_model = coremltools.converters.xgboost.convert(booster, feature_names)
        model_path = tmp_folder + "model.mlmodel"
        cml_model.save(model_path)
        output_tar.add(model_path, arcname="model.mlmodel")

    with open(output_path, 'rb') as output:
      output_buf = output.read()

  finally:
    shutil.rmtree(tmp_folder, ignore_errors=True)

  return output_buf



# Tests
# Call with 3 arguments:
# 1) xgboost .tar.gz arhive file path, should contain "xgboost-model" file,
# without extension
# 2) JSON metadata path
# 3) Output folder (optional)
#import sys
#import os
#
#with open(sys.argv[1], 'rb') as xgb_model_file, \
#     open(sys.argv[2], 'r') as metadata_file:
#
#  xgb_model_buf = xgb_model_file.read()
#  metadata = json.load(metadata_file)
#  output_tar_buf = transform_model(xgb_model_buf, metadata)
#
#  out_path = (sys.argv[3] if len(sys.argv) >= 4 else os.getcwd()) + "/out.tar.gz"
#  print("Output:", out_path)
#  print("Bytes:", len(output_tar_buf))
#
#  with open(out_path, 'wb') as output:
#    output.write(output_tar_buf)
