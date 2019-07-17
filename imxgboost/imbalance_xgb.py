import sys
sys.path.append("..")
import numpy as np
import xgboost as xgb
from imxgboost.weighted_loss import Weight_Binary_Cross_Entropy
from imxgboost.focal_loss import Focal_Binary_Loss
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

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

class imbalance_xgboost(BaseEstimator,ClassifierMixin):
    """Data in the form of [nData * nDim], where nDim stands for the number of features.
       This wrapper would provide a Xgboost interface with sklearn estimiator structure, which could be stacked in other Sk pipelines
    """
    def __init__(self,num_round=10,max_depth=10,eta=0.3,silent_mode=True,objective_func='binary:logitraw',eval_metric='logloss',booster='gbtree',special_objective=None, imbalance_alpha=None, focal_gamma=None):
        """
        Parameters to initialize a Xgboost estimator
        :param num_round. The rounds we would like to iterate to train the model
        :param max_depth. The maximum depth of the classification boosting, need to be specified
        :param num_class. The number of classes for the classifier
        :param eta Step. Size shrinkage used in update to prevents overfitting
        :param silent_mode. Set to 'True' or 'False' to determine if print the information during training. True is higly recommended
        :param objective_func. The objective function we would like to optimize
        :param eval_metric. The loss metrix. Note this is partially correlated to the objective function, and unfit loss function would lead to problematic loss
        :param booster. The booster to be usde, can be 'gbtree', 'gblinear' or 'dart'.
        :param imbalance_alpha. The \alpha value for imbalanced loss. Will make impact on '1' classes. Must have when special_objective 'weighted'
        :param focal_gamma. The \gamma value for focal loss. Must have when special_objective 'focal'
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
        self.imbalance_alpha = imbalance_alpha
        self.focal_gamma = focal_gamma


    def fit(self,data_x,data_y):
        if self.special_objective is None:
            # get the parameter list
            self.para_dict = {'max_depth': self.max_depth,
                              'eta': self.eta, 
                              'silent': self.silent_mode, 
                              'objective': self.objective_func,
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
            # if the alpha value is None then raise an error
            if self.imbalance_alpha is None:
                raise ValueError('Argument imbalance_alpha must have a value when the objective is \'weighted\'!')
            # construct the object with imbalanced alpha value
            weighted_loss_obj = Weight_Binary_Cross_Entropy(imbalance_alpha=self.imbalance_alpha)
            # fit the classfifier
            self.boosting_model = xgb.train(self.para_dict, dtrain, self.num_round, self.eval_list, obj=weighted_loss_obj.weighted_binary_cross_entropy, feval=evalerror, verbose_eval=False)
        elif self.special_objective == 'focal':
            # if the gamma value is None then raise an error
            if self.focal_gamma is None:
                raise ValueError('Argument focal_gamma must have a value when the objective is \'focal\'!')
            # construct the object with focal gamma value
            focal_loss_obj = Focal_Binary_Loss(gamma_indct=self.focal_gamma)
            # fit the classfifier
            self.boosting_model = xgb.train(self.para_dict, dtrain, self.num_round, self.eval_list, obj=focal_loss_obj.focal_binary_object, feval=evalerror, verbose_eval=False)
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

    def predict_sigmoid(self,data_x, y=None):
        # sigmoid output, for the prob = 1

        raw_output = self.predict(data_x, y)
        sigmoid_output = 1. / (1. + np.exp(-raw_output))

        return sigmoid_output
    
    def predict_determine(self,data_x,y=None):
        # deterministic output
        sigmoid_output = self.predict_sigmoid(data_x, y)
        prediction_output = np.round(sigmoid_output)
        
        return prediction_output

    def predict_two_class(self, data_x, y=None):
        # predict the probability of two classes
        prediction_output = two_class_encoding(self.predict(data_x, y))

        return prediction_output
    
    def score_func(self, y_true, y_pred, mode='accuracy'):
        prob_pred = two_class_encoding(y_pred)
        label_pred = np.argmax(prob_pred, axis=1)
        if mode=='accuracy':
            score_pred = accuracy_score(y_true=y_true, y_pred=label_pred)
        elif mode == 'precision':
            score_pred = precision_score(y_true=y_true, y_pred=label_pred)
        elif mode == 'recall':
            score_pred = recall_score(y_true=y_true, y_pred=label_pred)
        elif mode == 'f1':
            score_pred = f1_score(y_true=y_true, y_pred=label_pred)
        elif mode == 'MCC':
            score_pred = matthews_corrcoef(y_true=y_true, y_pred=label_pred)
        else:
            raise ValueError('Score function mode unrecognized! Must from one in the list '
                             '[\'accuracy\', \'precision\',\'recall\',\'f1\',\'MCC\']')
        
        return score_pred