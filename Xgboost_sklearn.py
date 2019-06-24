import time
import math
import scipy
import random
import numpy as np
import xgboost as xgb
from numpy import matlib
import sklearn as sk
from sklearn import linear_model
#import sklearn-based class to perform pipeline process
from sklearn.base import BaseEstimator, ClassifierMixin
import collections

def weighted_binary_cross_entropy(pred,dtrain,imbalance_alpha=6.5):
    # retrieve data from dtrain matrix
    label = dtrain.get_label()
    # compute the prediction with sigmoid
    sigmoid_pred = 1.0 / (1.0 + np.exp(-pred))
    # gradient
    grad = -(imbalance_alpha**label)*(label - sigmoid_pred)
    hess = (imbalance_alpha**label)*sigmoid_pred*(1.0 - sigmoid_pred)
    
    return grad, hess

def robust_pow(num_base, num_pow):
	# numpy does not permit negative numbers to fractional power
	# use this to perform the power algorithmic
	return np.sign(num_base) * (np.abs(num_base)) ** (num_pow)

def focal_binary_object(pred,dtrain,gamma_indct=3.0):
    # retrieve data from dtrain matrix
    label = dtrain.get_label()
    # compute the prediction with sigmoid
    sigmoid_pred = 1.0 / (1.0 + np.exp(-pred))
    # gradient
    # complex gradient with different parts
    g1 = sigmoid_pred * (1-sigmoid_pred) 
    g2 = label+((-1)**label)*sigmoid_pred
    g3 = sigmoid_pred + label - 1
    g4 = 1 - label - ((-1)**label)*sigmoid_pred
    g5 = label + ((-1)**label)*sigmoid_pred
    # combine the gradient
    grad = gamma_indct*g3*robust_pow(g2, gamma_indct)*np.log(g4+1e-9) + ((-1)**label)*robust_pow(g5, (gamma_indct+1))
    # combine the gradient parts to get hessian components
    hess_1 = robust_pow(g2, gamma_indct) + gamma_indct*((-1)**label)*g3*robust_pow(g2, (gamma_indct-1))
    hess_2 = ((-1)**label)*g3*robust_pow(g2, gamma_indct)/g4
    # get the final 2nd order derivative
    hess = ((hess_1*np.log(g4+1e-9)-hess_2)*gamma_indct + (gamma_indct+1)*robust_pow(g5, gamma_indct))*g1
    
    return grad, hess

def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    # return a pair metric_name, result
    # since preds are margin(before logistic transformation, cutoff at 0)
    
    return 'error', float(sum(labels != (preds > 0.0))) / len(labels)

def two_class_encoding(flat_prediction):
    if len(np.shape(flat_prediction))==2:
        return flat_prediction
    else:
        # class 1 probability
        class_one_prob = 1.0 / (1.0 + np.exp(-flat_prediction))
        class_one_prob = np.reshape(class_one_prob,[-1,1])
        # class 0 probability
        class_zero_prob = 1 - class_one_prob
        class_zero_prob = np.reshape(class_zero_prob,[-1,1])
        # concatenate the probabilities to get the final prediction
        sigmoid_two_class_pred = np.concatenate((class_zero_prob,class_one_prob),axis=1)
    
        return sigmoid_two_class_pred

class Xgboost_classsifier_sklearn(BaseEstimator,ClassifierMixin):
    """Data in the form of [nData * nDim], where nDim stands for the number of features.
       This wrapper would provide a Xgboost interface with sklearn estimiator structure, which could be stacked in other Sk pipelines
    """
    def __init__(self,num_round=10,max_depth=10,eta=0.3,silent_mode=True,objective_func='multi:softprob',eval_metric='mlogloss',booster='gbtree',special_objective=None):
        """
        Parameters to initialize a Xgboost estimator
        @param: num_round. The rounds we would like to iterate to train the model
        @param: max_depth. The maximum depth of the classification boosting, need to be specified
        @param: num_class. The number of classes for the classifier
        @param: eta Step. Size shrinkage used in update to prevents overfitting
        @param: silent_mode. Set to 'True' or 'False' to determine if print the information during training. True is higly recommended
        @param: objective_func. The objective function we would like to optimize
        @param: eval_metric. The loss metrix. Note this is partially correlated to the objective function, and unfit loss function would lead to problematic loss
        @param: booster. The booster to be usde, can be 'gbtree', 'gblinear' or 'dart'.
        """
        self.num_round = num_round
        self.max_depth = max_depth
        self.eta = eta
        self.silent_mode = silent_mode
        self.objective_func = objective_func
        self.eval_metric = eval_metric
        self.booster = booster
        self.eval_list = []
        self.boosting_model = 0
        self.special_objective = special_objective


    def fit(self,data_x,data_y,num_class):
        if self.special_objective is None:
            # get the parameter list
            self.para_dict = {'max_depth': self.max_depth,
                              'eta': self.eta, 
                              'silent': self.silent_mode, 
                              'objective': self.objective_func, 
                              'num_class': num_class,
                              'eval_metric': self.eval_metric,
                              'booster': self.booster}
        else:
            # get the parameter list, without stating the objective function
            self.para_dict = {'max_depth': self.max_depth,
                              'eta': self.eta, 
                              'silent': self.silent_mode,
                              'eval_metric': self.eval_metric,
                              'booster': self.booster}
        # make sure data is in [nData * nSample] format
        assert len(data_x.shape)==2
        # check if data length is the same
        if data_x.shape[0]!=data_y.shape[0]:
            raise ValueError('The numbner of instances for x and y data should be the same!')
        # data_x is in [nData*nDim]
        nData = data_x.shape[0]
        nDim = data_x.shape[1]
        # split the data into train and validation
        holistic_ind = np.random.permutation(nData)
        train_ind = holistic_ind[0:nData*3//4]
        valid_ind = holistic_ind[nData*3//4:nData]
        # indexing and get the data
        train_data = data_x[train_ind]
        train_label = data_y[train_ind]
        valid_data = data_x[valid_ind]
        valid_label = data_y[valid_ind]
        # marixilize the data and train the estimator
        dtrain = xgb.DMatrix(train_data, label=train_label)
        dvalid = xgb.DMatrix(valid_data, label=valid_label)
        self.eval_list = [(dvalid, 'valid'), (dtrain, 'train')]
        if self.special_objective is None:
            # fit the classfifier
            self.boosting_model = xgb.train(self.para_dict, dtrain, self.num_round, self.eval_list, verbose_eval=False)
        elif self.special_objective == 'weighted':
            # fit the classfifier
            self.boosting_model = xgb.train(self.para_dict, dtrain, self.num_round, self.eval_list, weighted_binary_cross_entropy, evalerror, verbose_eval=False)
        elif self.special_objective == 'focal':
            # fit the classfifier
            self.boosting_model = xgb.train(self.para_dict, dtrain, self.num_round, self.eval_list, focal_binary_object, evalerror, verbose_eval=False)
        else:
            raise ValueError('The input special objective mode not recognized! Could only be \'weighted\' or \'focal\', but got '+str(self.special_objective))

    def predict(self,data_x,y=None):
        # matrixilize
        if y is not None:
            try:
                dtest = xgb.DMatrix(data_x,label=y)
            except:
                raise ValueError('Test data invalid!')
        else:
            dtest = xgb.DMatrix(data_x)
        
        prediction_output = self.boosting_model.predict(dtest)
        
        return prediction_output
    
    def predict_determine(self,data_x,y=None):
        # deterministic output
        if y is not None:
            try:
                dtest = xgb.DMatrix(data_x,label=y)
            except:
                raise ValueError('Test data invalid!')
        else:
            dtest = xgb.DMatrix(data_x)
        
        raw_output = self.boosting_model.predict(dtest)
        sigmoid_output = 1. / (1. + np.exp(-raw_output))
        prediction_output = np.round(sigmoid_output)
        
        return prediction_output

    def predict_determine_v2(self,data_x,y=None):
        # deterministic output
        if y is not None:
            try:
                dtest = xgb.DMatrix(data_x,label=y)
            except:
                raise ValueError('Test data invalid!')
        else:
            dtest = xgb.DMatrix(data_x)
        
        prediction_output = np.argmax(self.boosting_model.predict(dtest),axis=1)
        
        return prediction_output
    
    def score(self, data_x, y):
        prob_pred = two_class_encoding(self.predict(data_x))
        label_pred = np.argmax(prob_pred,axis=1)
        accu_pred = np.sum(np.equal(label_pred,y))/label_pred.shape[0]
        
        return accu_pred