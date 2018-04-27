import numpy as np
from sklearn import metrics

def score_thres(y_true, y_pred_probas, threshold = 0.5, method='avg'):
    """ Function to evaluate the model with various metrics with a threshold value
    INPUTS
    ------
    y_true : true label
    y_pred_probas : predicted probabilities at each entry  
    threshold : threshold value of being classfied as 1
    OUTPUTS
    -------
    metric score
    """
    y_score = np.where(y_pred_probas >= threshold, 1, 0)
    if method == 'avg':
        return np.mean(np.mean(y_true == y_score, axis=1))
    elif method == 'exact':
        return metrics.accuracy_score(y_true, y_score)
    elif method == 'recall':
        return np.mean([metrics.recall_score(y_true[:,i], y_score[:,i]) for i in range(y_score.shape[1])])
    elif method == 'precision':
        return np.mean([metrics.precision_score(y_true[:,i], y_score[:,i]) for i in range(y_score.shape[1])])

def score(y_true, y_score, method='avg'):
    if method == 'avg':
        return np.mean(np.mean(y_true == y_score, axis=1))
    elif method == 'exact':
        return metrics.accuracy_score(y_true, y_score)
    elif method == 'recall':
        return np.mean([metrics.recall_score(y_true[:,i], y_score[:,i]) for i in range(y_score.shape[1])])
    elif method == 'precision':
        return np.mean([metrics.precision_score(y_true[:,i], y_score[:,i]) for i in range(y_score.shape[1])])