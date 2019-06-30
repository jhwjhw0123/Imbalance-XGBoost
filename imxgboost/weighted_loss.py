import numpy as np

class Weight_Binary_Cross_Entropy:
    '''
    The class of binary cross entropy loss, allows the users to change the weight parameter
    '''

    def __init__(self, imbalance_alpha):
        '''
        :param imbalance_alpha: the imbalanced \alpha value for the minority class (label as '1')
        '''
        self.imbalance_alpha = imbalance_alpha

    def weighted_binary_cross_entropy(self, pred, dtrain):
        # assign the value of imbalanced alpha
        imbalance_alpha = self.imbalance_alpha
        # retrieve data from dtrain matrix
        label = dtrain.get_label()
        # compute the prediction with sigmoid
        sigmoid_pred = 1.0 / (1.0 + np.exp(-pred))
        # gradient
        grad = -(imbalance_alpha ** label) * (label - sigmoid_pred)
        hess = (imbalance_alpha ** label) * sigmoid_pred * (1.0 - sigmoid_pred)

        return grad, hess