```python
from IPython.display import IFrame
```

# Propensity model results

## Synthetic variants creation

In order to create synthetic variants package [coolname](https://pypi.org/project/coolname/) was used. Package is able to generate 'meaningful' names which can be 2 - 5 words long. In order to make sure names would get non-zero features after feature encoding step the following strings must have been present in the generated name:

```python
 DES_WORDS = \
        ['simple', 'free', 'love', 'embrace', 'moment', 'gratitude', 'grateful',
         'fixed', 'live', 'now', 'hard', 'together', 'kind']

```

In order to make sure encoding creates more than one feature each synthetic varaint looked as follows:

```python

randomly_generated_name = 'grateful whale of much wisdom'

variant = {
    'text': randomly_generated_name,
    'chars': len(curr_text), 
    'words': len(randomly_generated_name.split(' '))}

```

300 of such dictionaries were generated as synthetic variants
   

## Propensities 

3 distributions were used to 'hardcode' variants' propensities:
 - uniform distribution (each variant had a chance of being chosen equal to `p = 1/N` where N is number of variants
 - normal with mean = 150 and SD = 30 (10000 samples were used to aproximate this distribution)
 - 1-param [Weibull](https://numpy.org/doc/stable/reference/random/generated/numpy.random.weibull.html) with a = 10 (10000 samples were used to aproximate this distribution)


```python
IFrame(src='./plots/props.html', width=1000, height=500)
```





<iframe
    width="1000"
    height="500"
    src="./plots/props.html"
    frameborder="0"
    allowfullscreen
></iframe>




## Cases / propensity model data creation ways:

 - #1 - for each decision all variants but chosen one get 0 and a chosen one gets 1. This approach creates N rows for each decision / sample (where `N = number of variants`):

 
 | variant          | is chosen | variant's weight |
 |------------------|-----------|----------------|
 | randomly selected best variant  | 1 | 1 |
 | 1st of remaining variants  | 0 | 1 |
 | 2nd of remaining variants  | 0 | 1 |
 | ...  | ... | ... |
 | last of remaining variants  | 0 | 1 |
 
 
 
 - #2 - chosen variant is flagged as 1 and gets weight of 1. one of not chosen variants is selected randomly and flagged as 0 with `weight =  N - 1` where `N = number of vairants`. This approach creates 2 rows per decision / sample:
 
 
  | variant          | is chosen | variant's weight |
 |------------------|-----------|----------------|
 | randomly selected best variant  | 1 | 1 |
 | one of remaining variants  | 0 | N - 1 |
 
 
  - #3 All variants have `is chosen` initially set to 0. When decision / sampling is made chosen variant is appended to variants' list with `is chosen` set to 1 (this makes chosen variant occur twice per decision / sample). To account for this effect propensity calculated by model must be corrected by: p<sub>corrected</sub> = p<sub>model</sub> / (1 - p<sub>model</sub>). This approach creates N + 1 rows per decisions / sample (where N is a number of variants)
 
  
 
 | variant          | is chosen | variant's weight |
 |------------------|-----------|----------------|
 | randomly selected best variant  | 1 | 1 |
 | randomly selected best variant  | 0 | 1 |
 | 1st of remaining variants  | 0 | 1 |
 | 2nd of remaining variants  | 0 | 1 |
 | ...  | ... | ... |
 | last of remaining variants  | 0 | 1 |
 

## Model

Vanilla XGBoost classifier was used as a propensity estimator. Variables obtained with feature encoding were used as predictor variables and `is chosen` flag was used a target variable. Feature encoding was performed with 'hash table' of the most recent messages model - [link to model](https://improve-v5-resources-prod-models-117097735164.s3-us-west-2.amazonaws.com/models/mindful/latest/improve-messages-2.0.xgb.gz).

# Results of propensity models


Case #2 has data for 5k+ decision / samples because data sets had signifficantly less rows and fit in my pc's RAM while #1 and #3 didn't (e.g. #2 had 10k rows for 5k decisions while #1 generated 1500000 rows for 300 variants and 5k decisions)

## Comparison of sum of obtained propensities for approaches #1, #2 and #3 and different distributions


```python
IFrame(src='./plots/model_propensity_sum-uni.html', width=1000, height=500)
```





<iframe
    width="1000"
    height="500"
    src="./plots/model_propensity_sum-uni.html"
    frameborder="0"
    allowfullscreen
></iframe>





```python
IFrame(src='./plots/model_propensity_sum-norm_m150_sd30.html', width=1000, height=500)
```





<iframe
    width="1000"
    height="500"
    src="./plots/model_propensity_sum-norm_m150_sd30.html"
    frameborder="0"
    allowfullscreen
></iframe>





```python
IFrame(src='./plots/model_propensity_sum-weib_rl5_a10.html', width=1000, height=500)
```





<iframe
    width="1000"
    height="500"
    src="./plots/model_propensity_sum-weib_rl5_a10.html"
    frameborder="0"
    allowfullscreen
></iframe>




## Comparison of Symmetric Mean Absolute Percentage Error (SMAPE) of obtained propensities vs hardcoded propensities for approaches #1, #2 and #3 and different distributions


```python
IFrame(src='./plots/propensity_smape-uni.html', width=1000, height=500)
```





<iframe
    width="1000"
    height="500"
    src="./plots/propensity_smape-uni.html"
    frameborder="0"
    allowfullscreen
></iframe>





```python
IFrame(src='./plots/propensity_smape-norm_m150_sd30.html', width=1000, height=500)
```





<iframe
    width="1000"
    height="500"
    src="./plots/propensity_smape-norm_m150_sd30.html"
    frameborder="0"
    allowfullscreen
></iframe>





```python
IFrame(src='./plots/propensity_smape-weib_rl5_a10.html', width=1000, height=500)
```





<iframe
    width="1000"
    height="500"
    src="./plots/propensity_smape-weib_rl5_a10.html"
    frameborder="0"
    allowfullscreen
></iframe>




## Comparison of total computation time (4 cores) for approaches #1, #2 and #3 and different distributions


```python
IFrame(src='./plots/total_duration_mins-uni.html', width=1000, height=500)
```





<iframe
    width="1000"
    height="500"
    src="./plots/total_duration_mins-uni.html"
    frameborder="0"
    allowfullscreen
></iframe>





```python
IFrame(src='./plots/total_duration_mins-norm_m150_sd30.html', width=1000, height=500)
```





<iframe
    width="1000"
    height="500"
    src="./plots/total_duration_mins-norm_m150_sd30.html"
    frameborder="0"
    allowfullscreen
></iframe>





```python
IFrame(src='./plots/total_duration_mins-weib_rl5_a10.html', width=1000, height=500)
```





<iframe
    width="1000"
    height="500"
    src="./plots/total_duration_mins-weib_rl5_a10.html"
    frameborder="0"
    allowfullscreen
></iframe>



