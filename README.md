# Imbalance-XGBoost
This software includes the codes of Weighted Loss and Focal Loss [1] implementation for [XGBoost](https://github.com/dmlc/xgboost) [2] in binary classification problems. The principal reason for us to use Weighted and Focal Loss functions is to address the problem of label-imbalanced data. The original XGBoost program provides a convenient way to customize the loss function, but one needs to compute the first and second order derivatives to implement them. The major contribution of the software is the derivation of the gradients and the implementations of them.

<!-- ## Software Update
**The project has been posted on github for several months, and now a correponding API on Pypi is released. Special thanks to @icegrid and @shaojunchao for help correct errors in the previous versions. The codes are now updated to version 0.7 and it now allows users to specify the weighted parameter \alpha and focal parameter \gamma outside the script. Also it supports higher version of XGBoost now.** <br /> -->

## Software Update
**Version 0.8.1: The package now supports early stopping, you can specify this by `early_stopping_rounds` when initializing the object.**

## Version Notification
**From version 0.7.0 on Imbalance-XGBoost starts to support higher versions of XGBoost and removes supports of versions earlier than 0.4a30(XGBoost>=0.4a30). This contradicts the previous requirement of XGBoost<=0.4a30. Please choose the version that fits your system accordingly.** <br />
**Starting from version 0.8.1, the package now requires xgboost to have a newer version of >=1.1.1. This is due to some changes on deprecated arguments of the XGBoost.**

## Installation
Installing with Pypi is the easiest way, you can run: <br />

```
pip install imbalance-xgboost
```
If you have multiple versions of Python, make sure you're using Python 3 (run with `pip3 install imbalance-xgboost`). The program are designated for Python 3.5 and 3.6. That being said, an (incomplete) test does not find any compatible issue on Python 3.7 and 3.8. <br />

The package has a hard dependency on numpy, sklearn and xgboost. <br />

## Usage
To use the wrapper, one needs to import *imbalance_xgboost* from module **imxgboost.imbalance_xgb**. An example is given below: <br /> 

```Python
from imxgboost.imbalance_xgb import imbalance_xgboost as imb_xgb
```
The specific loss function could be set through *special_objective* parameter. Specifically, one could construct a booster with: <br />
```Python
xgboster = imb_xgb(special_objective='focal')
```
for *focal loss* and <br />
```Python
xgboster = imb_xgb(special_objective='weighted')
```
for *weighted* loss. The parameters $\alpha$ and $\gamma$ can be specified by giving a value when constructing the object. In addition, the class is designed to be compatible with scikit-learn package, and you can treat it as a sk-learn classifier object. Thus, it will be easy to use methods in Sklearn such as *GridsearchCV* to perform grid search for the parameters of focal and weighted loss functions. <br />
```Python
from sklearn.model_selection import GridSearchCV
xgboster_focal = imb_xgb(special_objective='focal')
xgboster_weight = imb_xgb(special_objective='weighted')
CV_focal_booster = GridSearchCV(xgboster_focal, {"focal_gamma":[1.0,1.5,2.0,2.5,3.0]})
CV_weight_booster = GridSearchCV(xgboster_weight, {"imbalance_alpha":[1.5,2.0,2.5,3.0,4.0]})
```
The data fed to the booster should be of numpy type and following the convention of: <br />
x: [nData, nDim] <br />
y: [nData,] <br />
In other words, the x_input should be row-major and labels should be flat. <br />
And finally, one could fit the data with Cross-validation and retrieve the optimal model: <br />
```Python
CV_focal_booster.fit(records, labels)
CV_weight_booster.fit(records, labels)
opt_focal_booster = CV_focal_booster.best_estimator_
opt_weight_booster = CV_weight_booster.best_estimator_
```
After getting the optimal booster, one will be able to make predictions. There are following methods to make predictions with imabalnce-xgboost: <br />
Method `predict` <br />
```Python
raw_output = opt_focal_booster.predict(data_x, y=None) 
```
This will return the value of '$z_i$' before applying sigmoid.  <br />
Method `predict_sigmoid` <br />
```Python
sigmoid_output = opt_focal_booster.predict_sigmoid(data_x, y=None) 
```
This will return the $\hat{y}$ value, which is $p(y=1|x)$ for 2-class classification.  <br />
Method `predict_determine` <br />
```Python
class_output = opt_focal_booster.predict_determine(data_x, y=None) 
```
This will return the predicted logit, which 0 or 1 in the 2-class scenario.  <br />
Method `predict_two_class` <br />
```Python
prob_output = opt_focal_booster.predict_two_class(data_x, y=None) 
```
This will return the predicted probability of 2 classes, in the form of [nData * 2]. The first column is the probability of classifying the data point to 0 and the second column is the prob of classifying as 1. <br />
To assist the evaluation of classification results, the package provides a score function `score_eval_func()` with multiple metrics. One can use `make_scorer()` method in sk-learn and `functools` to specify the evaluation score. The method will be compatible with sk-learn cross validation and model selection processes. <br />
```Python
import functools
from sklearn.metrics import make_scorer
from sklearn.model_selection import LeaveOneOut, cross_validate
# retrieve the best parameters
xgboost_opt_param = CV_focal_booster.best_params_
# instantialize an imbalance-xgboost instance
xgboost_opt = imb_xgb(special_objective='focal', **xgboost_opt_param)
# cross-validation
# initialize the splitter
loo_splitter = LeaveOneOut()
# initialize the score evalutation function by feeding the 'mode' argument
# 'mode' can be [\'accuracy\', \'precision\',\'recall\',\'f1\',\'MCC\']
score_eval_func = functools.partial(xgboost_opt.score_eval_func, mode='accuracy')
# Leave-One cross validation
loo_info_dict = cross_validate(xgboost_opt, X=x, y=y, cv=loo_splitter, scoring=make_scorer(score_eval_func))
```
In the new version, we can also collect the information of the confusion matrix through the `correct_eval_func` provided. This enables the users to evluate the metrics like accuracy, precision, and recall for the average/overall test sets in the cross-validation process. <br />
```Python
# initialize the correctness evalutation function by feeding the 'mode' argument
# 'mode' can be ['TP', 'TN', 'FP', 'FN']
TP_eval_func = functools.partial(xgboost_opt.score_eval_func, mode='TP')
TN_eval_func = functools.partial(xgboost_opt.score_eval_func, mode='FP')
FP_eval_func = functools.partial(xgboost_opt.score_eval_func, mode='TN')
FN_eval_func = functools.partial(xgboost_opt.score_eval_func, mode='FN')
# define the score function dictionary
score_dict = {'TP': make_scorer(TP_eval_func), 
              'FP': make_scorer(TN_eval_func), 
              'TN': make_scorer(FP_eval_func), 
              'FN': make_scorer(FN_eval_func)}
# Leave-One cross validation
loo_info_dict = cross_validate(xgboost_opt, X=x, y=y, cv=loo_splitter, scoring=score_dict)
overall_tp = np.sum(loo_info_dict['test_TP']).astype('float')
```
More soring function may be added in later versions.

## Theories and derivatives
You don't have to understand the equations if you find they are hard to grasp, you can simply use it with the API. However, for the purpose of understanding, the derivatives of the two loss functions are listed. <br />
For both loss functions, since the task is 2-class classification, the activation would be sigmoid:
$$y_i = \frac{1}{ 1 + \text{exp}(-z_i)} $$
Below the two types of loss will be discussed respectively.
### 1. Weighted Imbalance (Cross-entropoy) Loss
Let $\hat{y}$ denote the true labels. The weighted imbalance loss for 2-class data can be denoted as:

$$ L_{w} = -\sum_{i=1}^m(\alpha\hat{y}_i\log(y_i) + (1-\hat{y}_i)\log(1-y_i)) $$

where $\alpha$ is the 'imbalance factor'. If $\alpha$ is greater than 1, then we put extra loss on 'classifying 1 as 0'.<br />
The gradient is:

$$\frac{\partial L_{w}}{\partial z_{i}} = -\alpha^{\hat{y}_i}(\hat{y}_i -y_i)$$

Moreover, the second order gradient correspond to
$$\frac{\partial L_w^2}{\partial^{2} z_{i}} = \alpha^{\hat{y}_i}(1-y_i)(y_i)$$
### 2. Focal Loss
According to [1], the focal loss can be denoted as 

$$ L_{f} = - \sum_{i=1}^m \hat{y}_i (1 - y_i)^\gamma \log(1 - \hat{y}_i - (-1)^{\hat{y}_i}y_i) + (1 - \hat{y}_i) y^{\gamma}_i \log(1- y_i) $$

The first order gradient is

$$ \frac{\partial L_{f}}{\partial z_i} = 
\gamma\left[(y_i + \hat{y}_i - 1) (\hat{y}_i + (-1)^{ \hat{y}_i}y_i)^\gamma \log(1 - \hat{y}_i - (-1)^{\hat{y}_i}y_i) \right] + (-1)^{\hat{y}_i}(\hat{y}_i +(-1)^{\hat{y}_i}y_i)^{\gamma + 1}
$$

To simplify the expression above, one can set the following short-hand variables:

$$ \begin{cases}
\eta_1 =  y_i(1-y_i) \\
\eta_2 = \hat{y}_i + (-1)^{\hat{y}_i}y_i\\
\eta_3 = y_i + \hat{y}_i - 1 \\
\eta_4 = 1 - \hat{y}_i - (-1)^{\hat{y}_i}y_i
\end{cases}$$

Using the above notations, the 1-st order derivative can be written as 
$$\frac{\partial L_f}{\partial z_i} = \gamma\eta_3\eta_2^\gamma \log(\eta_4) + (-1)^{\hat{y}_i}\eta_2^{\gamma + 1}$$ 
Finally, the 2-nd order derivative is
$$\frac{\partial^{2} L_f}{\partial z_i^2} = \eta_1 \left[ \gamma \left(\eta_2^\gamma + \gamma(-1)^{\hat{y}_i}\eta_3\eta_2^{\gamma - 1}\log(\eta_4) - \frac{(-1)^{\hat{y}_i}\eta_3 \eta_2^{\gamma}}{\eta_4}\right) + (\gamma + 1) \eta_2^\gamma\right]
$$

## Paper Citation
If you use this package in your research please cite our paper: <br />
```
@misc{wang2019imbalancexgboost,
    title={Imbalance-XGBoost: Leveraging Weighted and Focal Losses for Binary Label-Imbalanced Classification with XGBoost},
    author={Chen Wang and Chengyuan Deng and Suzhen Wang},
    year={2019},
    eprint={1908.01672},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
**More information on the software**: <br />
@author: Chen Wang, Dept. of Computer Science, School of Art and Science, Rutgers University (previously affiliated with University College London, Sichuan University and Northwestern Polytechnical University) <br/>
@version: 0.8.1

## References
[1] Lin, Tsung-Yi, Priyal Goyal, Ross Girshick, Kaiming He, and Piotr Dollár. "Focal loss for dense object detection." IEEE transactions on pattern analysis and machine intelligence (2018). <br/>
[2] Chen, Tianqi, and Carlos Guestrin. "Xgboost: A scalable tree boosting system." In Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining, pp. 785-794. ACM, 2016.
