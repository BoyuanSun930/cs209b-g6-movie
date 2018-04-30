import numpy as np
from sklearn import metrics
import pandas as pd

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
    valid_metrics = {'avg', 'exact', 'recall', 'precision', 'hit', 'f1'}
    if method in valid_metrics:
        y_score = np.where(y_pred_probas >= threshold, 1, 0)
        if method == 'avg':
            return np.mean(np.mean(y_true == y_score, axis=1))
        elif method == 'exact':
            return metrics.accuracy_score(y_true, y_score)
        elif method == 'recall':
            return np.mean([metrics.recall_score(y_true[:,i], y_score[:,i]) for i in range(y_score.shape[1])])
        elif method == 'precision':
            return np.mean([metrics.precision_score(y_true[:,i], y_score[:,i]) for i in range(y_score.shape[1])])
        elif method == 'hit':
            y_total = y_true + y_score
            hit_rate = np.sum([(y_total[i] == 2).any() for i in range(len(y_total))])
            return hit_rate / len(y_total)
        else:
            recall = np.mean([metrics.recall_score(y_true[:,i], y_score[:,i]) for i in range(y_score.shape[1])])
            precision = np.mean([metrics.precision_score(y_true[:,i], y_score[:,i]) for i in range(y_score.shape[1])])
            return 2 * recall * precision / (recall + precision)
    else:
        raise ValueError('Invalid Metric')

def score(y_true, y_score, method='avg'):
    """ Function to evaluate the model with various metrics with a threshold value
        INPUTS
        ------
        y_true : true label
        y_score : predicted label
        threshold : threshold value of being classfied as 1
        OUTPUTS
        -------
        metric score
        """
    valid_metrics = {'avg', 'exact', 'recall', 'precision', 'hit', 'f1'}
    if method in valid_metrics:
        if method == 'avg':
            return np.mean(np.mean(y_true == y_score, axis=1))
        elif method == 'exact':
            return metrics.accuracy_score(y_true, y_score)
        elif method == 'recall':
            return np.mean([metrics.recall_score(y_true[:,i], y_score[:,i]) for i in range(y_score.shape[1])])
        elif method == 'precision':
            return np.mean([metrics.precision_score(y_true[:,i], y_score[:,i]) for i in range(y_score.shape[1])])
        elif method == 'hit':
            y_total = y_true + y_score
            hit_rate = np.sum([(y_total[i] == 2).any() for i in range(len(y_total))])
            return hit_rate / len(y_total)
        else:
            recall = np.mean([metrics.recall_score(y_true[:,i], y_score[:,i]) for i in range(y_score.shape[1])])
            precision = np.mean([metrics.precision_score(y_true[:,i], y_score[:,i]) for i in range(y_score.shape[1])])
            return 2 * recall * precision / (recall + precision)
    else:
        raise ValueError('Invalid Metric')
        
def metrics_df(y_train, y_test, train_pred, test_pred):
    train_avg = score(y_train, train_pred, 'avg')
    train_exact = score(y_train, train_pred, 'exact')
    train_prec = score(y_train, train_pred, 'precision')
    train_rec = score(y_train, train_pred, 'recall')
    train_hit = score(y_train, train_pred, 'hit')
    train_f1 = score(y_train, train_pred, 'f1')
    
    test_avg = score(y_test, test_pred, 'avg')
    test_exact = score(y_test, test_pred, 'exact')
    test_prec = score(y_test, test_pred, 'precision')
    test_rec = score(y_test, test_pred, 'recall')
    test_hit = score(y_test, test_pred, 'hit')
    test_f1 = score(y_test, test_pred, 'f1')

    return pd.Series([train_avg, train_exact, train_prec, train_rec, train_hit, train_f1, 
                      test_avg, test_exact, test_prec, test_rec, test_hit, test_f1],
                    index = ['Train Avg Accuracy', 'Train Exact Accuracy', 'Train Precision', 'Train Recall',
                             'Train Hit Rate', 'Train F1 Score', 'Test Avg Accuracy', "Test Exact Accuracy", 
                            'Test Precision', 'Test Recall','Test Hit Rate', 'Test F1 Score'])

def metrics_thres_df(y_train, y_test, train_pred, test_pred):
    train_avg = score_thres(y_train, train_pred, method = 'avg')
    train_exact = score_thres(y_train, train_pred, method ='exact')
    train_prec = score_thres(y_train, train_pred, method ='precision')
    train_rec = score_thres(y_train, train_pred, method ='recall')
    train_hit = score_thres(y_train, train_pred,method = 'hit')
    train_f1 = score_thres(y_train, train_pred, method ='f1')
    
    test_avg = score_thres(y_test, test_pred, method ='avg')
    test_exact = score_thres(y_test, test_pred, method ='exact')
    test_prec = score_thres(y_test, test_pred,method = 'precision')
    test_rec = score_thres(y_test, test_pred, method ='recall')
    test_hit = score_thres(y_test, test_pred, method ='hit')
    test_f1 = score_thres(y_test, test_pred, method ='f1')

    return pd.Series([train_avg, train_exact, train_prec, train_rec, train_hit, train_f1, 
                      test_avg, test_exact, test_prec, test_rec, test_hit, test_f1],
                    index = ['Train Avg Accuracy', 'Train Exact Accuracy', 'Train Precision', 'Train Recall',
                             'Train Hit Rate', 'Train F1 Score', 'Test Avg Accuracy', "Test Exact Accuracy", 
                            'Test Precision', 'Test Recall','Test Hit Rate', 'Test F1 Score'])

def plot_learning_curve(results, ax):
    """ Function to visualize the train / test learning curve from HW5
        
        INPUTS
        ------
        results : keras history
        ax : ax object
        
        OUTPUTS
        -------
        learning curve
        """
    ax.plot(results.history['acc'], label='Train')
    ax.plot(results.history['val_acc'], label='Test')
    ax.set_ylim([0,1])
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.legend(loc='best')

def genre_acc(y_true, y_pred, title, is_neural_net = False, thres = 0.5, genre_dict = genre_dict):
    """ Function to calculate and visualize the accuracy by genre.
        IMPORTANT NOTE: the predictions of neural network is prob, differet from sklearn classifiers,
        need to specify the classifier type
        INPUTS
        ------
        y_true : true labels
        y_pred : predicted label
        title : title of the plot
        genre_dict : index dictionary
        thres : threshold value for 1 if predicted by neural network
        
        OUTPUTS
        -------
        plot the by genre acc given the true labels and predicted label
        """
    
    if is_neural_net:
        y_pred = np.where(y_pred > thres, 1, 0)
    y_correct = ((y_true + y_pred) == 2).sum(0)
    y_total = (y_true == 1).sum(0)
    for index, key in enumerate(list(genre_dict.keys())[:20]):
        print('Genre {}: {} correct out of {}'.format(key, y_correct[index], y_total[index]))
    probas = y_correct / y_total
    plt.figure(figsize=(15,8))
    plt.bar(np.arange(20), probas, 0.5, alpha = 0.5, edgecolor = 'gold')
    plt.xticks(np.arange(20), list(genre_dict.keys()), rotation = 90, fontsize = 15)
    plt.ylabel('True Positive Accuracy', fontsize = 15)
    plt.title(title, fontsize = 15)