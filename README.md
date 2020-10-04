# python-sdk

### Supported formats:
 - *.mlmodel
 - *.xgb (gboost native format) 

### ImproveModel CLI
The prepared CLI takes as input:
 - one of supported method names to execute:
    - score - calculates predictions on all provided input data
    - sort - scores input and returns it ordered descendingly
    - choose - scores input and returns best choice along with score
 - desired model type to use out of 2 supported mdoel types:
    - *.mlmodel
    - *.xgb (xgboost native format)
 - path to desired model
 - JSON string with input data encapsulated with '' -> '<json string>'
 - JSON string with context encapsulated with '' -> '<json string>'
 - JSON string with model metadata encapsulated with '' -> '<json string>'

In order to use prepared ImproveModel CLI:
 - make sure to change directory to python-sdk folder
 - call improve_model_cli.py in the following fashion <br>
 python3.7 improve_model_cli.py [desired method name] [desired model type] [path to deired model] --variant [input JSON string] --context [context JSON string] --model_metadata [metadata JSON string]
 
To see example results please call: <br>
python3.7 improve_model_cli.py score xgb_native test_artifacts/model.xgb

To use CLI with files (i.e. for variants/context/model metadata/results) please use:
python3.7 improve_model_cli.py score xgb_native test_artifacts/model.xgb --context_pth test_artifacts/context.json --model_metadata_pth test_artifacts/model.json --results_pth test_artifacts/res.json --variants_pth test_artifacts/meditations.json 

### Results
Currently supported objectives:
 - regression - returns [input JSON string, score value, 0] for each observation
 - binary classification - returns [input JSON string, class 1 probability, class 1 label] for each observation
 - multiple classification - returns [input JSON string, highest class probability, most probable class label] for each observation

Results are always returned as a JSON strings: <br>
[[input JSON string, value, label], ...]
 - score method returns list of all inputs scored
 - sort method returns list of all inputs scored (if multiclass classification is the case then the list is sorted for each class from highest to lowest scores)
 - choose method returns best highest scored variant info ([input JSON string, value, label]). For binary classification best scores for class 1 are returned. For multiple classification best choices in each class are returned. Ties are broken randomly. 
   