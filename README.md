# Imbalance-Xgboost
This software includes the codes of Weighted Loss and Focal Loss [1] implementations for Xgboost [2](<\url> https://github.com/dmlc/xgboost) in binary classification problems. The principal reason for us to use Weighted and Focal Loss functions is to address the problem of label-imbalanced data. The original Xgboost program provides a convinient way to customize the loss function, but one will be needing to compute the first and second order derivatives to implement them. The major contribution of the software is the drivation of the gradients and the implementations of them. <br/>

## Software Release
**The project has been posted on github for several months, and now a correponding API on Pypi is released. Special thanks to @icegrid and @shaojunchao for help correct errors in the previous versions. The codes are now updated to version 0.6 and it now allows users to specify the weighted parameter \alpha and focal parameter \gamma outside the script.** <br />

## Installation
Installing with Pypi will be easiest way, you can run: <br />

```
pip install imbalance-xgboost
```
If you have multiple versions of Python, make sure you're using Python 3 (run with `pip3 install imbalance-xgboost`). Currently, the program only supports Python 3.5 and 3.6. <br />

The package has hard depedency on numpy, sklearn and xgboost. <br />

## Usage
To use the wrapper, one needs to import *imbalance_xgboost* from module **imxgboost.imbalance_xgb**. An example is given as bellow: <br /> 

```Python
from imxgboost.imbalance_xgb import imbalance_xgboost as imb_xgb
```
The specific loss function could be set through *special_objective* parameter. Specificly, one could construct a booster with: <br />
```Python
xgboster = imb_xgb(special_objective='focal')
```
for *focal loss* and <br />
```Python
xgboster = imb_xgb(special_objective='weighted')
```
for *weighted* loss. The prarameters $\alpha$ and $\gamma$ can be specified by giving a value when constructing the object. In addition, the class is designed to be compatible with scikit-learn package, and you can treat it as a sk-learn classifier object. Thus, it will be easy to use methods in Sklearn such as *GridsearchCV* to perform grid search for the parameters of focal and weighted loss functions. <br />
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
And finally, one could fit the data with Cross-validation and retreive the optimal model: <br />
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
This will return the value of 'zi' before applying sigmoid.  <br />
Method `predict_sigmoid` <br />
```Python
sigmoid_output = opt_focal_booster.predict_sigmoid(data_x, y=None) 
```
This will return the \hat{y} value, which is p(y=1|x) for 2-lcass classification.  <br />
Method `predict_determine` <br />
```Python
class_output = opt_focal_booster.predict_determine(data_x, y=None) 
```
This will return the predicted logit, which 0 or 1 in the 2-class scenario.  <br />
Method `predict_two_class` <br />
```Python
class_output = opt_focal_booster.predict_two_class(data_x, y=None) 
```
This will return the predicted probability of 2 classes, in the form of [nData * 2]. The first column is the probability of classifying the datapoint to 0 and the second column is the prob of classifying as 1. <br />

```
CV_weight_booster.fit(records, labels)
opt_focal_booster = CV_focal_booster.best_estimator_
opt_weight_booster = CV_weight_booster.best_estimator_
```

## Theories and derivatives
You don't have to understand the equations if you find they are hard to grasp, you can simply use it with the API. However, for the purpose of understanding, the derivatives of the two loss functions are listed. <br />
For both of the loss functions, since the task is 2-class classification, the activation would be sigmoid: <br />
<a href="https://www.codecogs.com/eqnedit.php?latex=y_{i}&space;=&space;\frac{1}{1&plus;\text{exp}(-z_{i})}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y_{i}&space;=&space;\frac{1}{1&plus;\text{exp}(-z_{i})}" title="y_{i} = \frac{1}{1+\text{exp}(-z_{i})}" /></a> <br />
And bellow the two types of loss will be discussed respectively. <br />
### 1. Weighted Imbalance (Cross-entropoy) Loss
And combining with $\hat{y}$, which are the true labels, the weighted imbalance loss for 2-class data could be denoted as: <br />
<a href="https://www.codecogs.com/eqnedit.php?latex=l_{w}&space;=&space;-\sum_{i=1}^{m}(\alpha\hat{y}_{i}\text{log}(y_{i})&plus;(1-\hat{y}_{i})\text{log}(1-y_{i})))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?l_{w}&space;=&space;-\sum_{i=1}^{m}(\alpha\hat{y}_{i}\text{log}(y_{i})&plus;(1-\hat{y}_{i})\text{log}(1-y_{i})))" title="l_{w} = -\sum_{i=1}^{m}(\alpha\hat{y}_{i}\text{log}(y_{i})+(1-\hat{y}_{i})\text{log}(1-y_{i})))" /></a>
<br />
Where $\alpha$ is the 'imbalance factor'. And $\alpha$ value greater than 1 means to put extra loss on 'classifying 1 as 0'.<br />
The gradient would be: <br />
<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;L_{w}}{\partial&space;z_{i}}&space;=&space;-\alpha^{\hat{y}_{i}}(\hat{y}_{i}-y_{i})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;L_{w}}{\partial&space;z_{i}}&space;=&space;-\alpha^{\hat{y}_{i}}(\hat{y}_{i}-y_{i})" title="\frac{\partial L_{w}}{\partial z_{i}} = -\alpha^{\hat{y}_{i}}(\hat{y}_{i}-y_{i})" /></a>  <br />
And the second order gradient would be: <br />
<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;L_{w}^{2}}{\partial^{2}&space;z_{i}}&space;=&space;\alpha^{\hat{y}_{i}}(1-y_{i})(y_{i})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;L_{w}^{2}}{\partial^{2}&space;z_{i}}&space;=&space;\alpha^{\hat{y}_{i}}(1-y_{i})(y_{i})" title="\frac{\partial L_{w}^{2}}{\partial^{2} z_{i}} = \alpha^{\hat{y}_{i}}(1-y_{i})(y_{i})" /></a>   <br />

### 2. Focal Loss
The focal loss is proposed in [1] and the expression of it would be: <br />
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;L_{w}&space;=&space;-\sum_{i=1}^{m}\hat{y}_{i}(1-y_{i})^{\gamma}\text{log}(y_{i})&space;&plus;&space;(1-\hat{y}_{i})y_{i}^{\gamma}\text{log}(1-y_{i})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;L_{w}&space;=&space;-\sum_{i=1}^{m}\hat{y}_{i}(1-y_{i})^{\gamma}\text{log}(y_{i})&space;&plus;&space;(1-\hat{y}_{i})y_{i}^{\gamma}\text{log}(1-y_{i})" title="L_{w} = -\sum_{i=1}^{m}\hat{y}_{i}(1-y_{i})^{\gamma}\text{log}(y_{i}) + (1-\hat{y}_{i})y_{i}^{\gamma}\text{log}(1-y_{i})" /></a> <br />
The first order gradient would be: <br />
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;\small&space;{\frac{\partial&space;L_{w}}{\partial&space;z_i}&space;=&space;\gamma(y_i&plus;\hat{y}_{i}-1)(\hat{y}_{i}&plus;(-1)^{\hat{y}}y_{i})^{\gamma}\text{log}(1-\hat{y}_{i}-(-1)^{\hat{y}_{i}}y_{i})&space;&plus;&space;(-1)^{\hat{y}_i}(\hat{y}_{i}&plus;(-1)^{\hat{y}_i}y_{i})^{\gamma&plus;1}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\small&space;{\frac{\partial&space;L_{w}}{\partial&space;z_i}&space;=&space;\gamma(y_i&plus;\hat{y}_{i}-1)(\hat{y}_{i}&plus;(-1)^{\hat{y}}y_{i})^{\gamma}\text{log}(1-\hat{y}_{i}-(-1)^{\hat{y}_{i}}y_{i})&space;&plus;&space;(-1)^{\hat{y}_i}(\hat{y}_{i}&plus;(-1)^{\hat{y}_i}y_{i})^{\gamma&plus;1}}" title="\small {\frac{\partial L_{w}}{\partial z_i} = \gamma(y_i+\hat{y}_{i}-1)(\hat{y}_{i}+(-1)^{\hat{y}}y_{i})^{\gamma}\text{log}(1-\hat{y}_{i}-(-1)^{\hat{y}_{i}}y_{i}) + (-1)^{\hat{y}_i}(\hat{y}_{i}+(-1)^{\hat{y}_i}y_{i})^{\gamma+1}}" /></a>    <br />
And the second order gradient would be a little bit complex. To simplify the expression, we firstly denotes the terms in the 1-st order gradient as the following notations: <br />
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;\begin{cases}&space;g_1&space;=&space;y_{i}(1-y_{i})\\&space;g_2=&space;\hat{y}_i&space;&plus;&space;(-1)^{\hat{y}_i}y_{i}\\&space;g_3&space;=&space;y_i&space;&plus;&space;\hat{y}_{i}-1\\&space;g_4&space;=&space;1-\hat{y}_i-(-1)^{\hat{y}_i}y_i\\&space;g_5&space;=&space;\hat{y}_i&space;&plus;&space;(-1)^{\hat{y}_i}&space;y_i&space;\end{cases}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\begin{cases}&space;g_1&space;=&space;y_{i}(1-y_{i})\\&space;g_2=&space;\hat{y}_i&space;&plus;&space;(-1)^{\hat{y}_i}y_{i}\\&space;g_3&space;=&space;y_i&space;&plus;&space;\hat{y}_{i}-1\\&space;g_4&space;=&space;1-\hat{y}_i-(-1)^{\hat{y}_i}y_i\\&space;g_5&space;=&space;\hat{y}_i&space;&plus;&space;(-1)^{\hat{y}_i}&space;y_i&space;\end{cases}" title="\begin{cases} g_1 = y_{i}(1-y_{i})\\ g_2= \hat{y}_i + (-1)^{\hat{y}_i}y_{i}\\ g_3 = y_i + \hat{y}_{i}-1\\ g_4 = 1-\hat{y}_i-(-1)^{\hat{y}_i}y_i\\ g_5 = \hat{y}_i + (-1)^{\hat{y}_i} y_i \end{cases}" /></a> <br />
Using the above notations, the 1-st order drivative will be: <br />
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;\large&space;\frac{\partial&space;L_w}{\partial&space;z_i}&space;=&space;\gamma&space;g_3&space;g_2^{\gamma}&space;\text{log}(g_4)&space;&plus;&space;(-1)^{\hat{y_i}}g_5^{\gamma&space;&plus;&space;1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\large&space;\frac{\partial&space;L_w}{\partial&space;z_i}&space;=&space;\gamma&space;g_3&space;g_2^{\gamma}&space;\text{log}(g_4)&space;&plus;&space;(-1)^{\hat{y_i}}g_5^{\gamma&space;&plus;&space;1}" title="\large \frac{\partial L_w}{\partial z_i} = \gamma g_3 g_2^{\gamma} \text{log}(g_4) + (-1)^{\hat{y_i}}g_5^{\gamma + 1}" /></a> <br />
Then the 2-nd order derivative will be: <br />
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;\frac{\partial^{2}&space;L}{\partial&space;z_{i}^{2}}&space;=&space;g_{1}\{\gamma[(g_2^{\gamma}&plus;\gamma&space;(-1)^{\hat{y}_{i}}g_3&space;g_2^{\gamma&space;-&space;1})\text{log}(g_4)-\frac{(-1)^{\hat{y}_i}g_3&space;g_2^{\gamma}}{g_4}]&space;&plus;&space;(\gamma&plus;1)g_5^{\gamma}\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\frac{\partial^{2}&space;L}{\partial&space;z_{i}^{2}}&space;=&space;g_{1}\{\gamma[(g_2^{\gamma}&plus;\gamma&space;(-1)^{\hat{y}_{i}}g_3&space;g_2^{\gamma&space;-&space;1})\text{log}(g_4)-\frac{(-1)^{\hat{y}_i}g_3&space;g_2^{\gamma}}{g_4}]&space;&plus;&space;(\gamma&plus;1)g_5^{\gamma}\}" title="\frac{\partial^{2} L}{\partial z_{i}^{2}} = g_{1}\{\gamma[(g_2^{\gamma}+\gamma (-1)^{\hat{y}_{i}}g_3 g_2^{\gamma - 1})\text{log}(g_4)-\frac{(-1)^{\hat{y}_i}g_3 g_2^{\gamma}}{g_4}] + (\gamma+1)g_5^{\gamma}\}" /></a>
<br />

## Enjoy Using!
@author: Chen Wang, Dept. of Computer Science, School of Art and Science, Rutgers University (previously affiliated with University College London, Sichuan University and Northwestern Polytechnical University) <br/>
@version: 0.6

## References
[1] Lin, Tsung-Yi, Priyal Goyal, Ross Girshick, Kaiming He, and Piotr Dollár. "Focal loss for dense object detection." IEEE transactions on pattern analysis and machine intelligence (2018). <br/>
[2] Chen, Tianqi, and Carlos Guestrin. "Xgboost: A scalable tree boosting system." In Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining, pp. 785-794. ACM, 2016.
