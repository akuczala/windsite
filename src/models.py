#from sklearn.linear_model import BayesianRidge
#from sklearn.linear_model import LinearRegression
#from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import explained_variance_score, r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from sklearn.metrics import f1_score, precision_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import normalize
from sklearn.inspection import permutation_importance, plot_partial_dependence
from sklearn.model_selection import train_test_split, ShuffleSplit, learning_curve
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import classification_report

import numpy as np
import matplotlib.pyplot as plt

def fit_regression(model,X,y,score = 'neg_root_mean_squared_error', random_state = 0):

    score_funs = {
    'explained_variance' : explained_variance_score,
    'r2' : r2_score, 'neg_root_mean_squared_error' : lambda *args: mean_squared_error(*args,squared=False)
    }
    score_fun = score_funs[score]

    #split the data into training / testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= random_state)

    #cross validate with f1 scores
    scores = cross_val_score(model,X_train,y_train,scoring = score)

    print('cross validation scores:', scores)

    #fit data and compare scores for training and testing data
    model = model.fit(X_train, y_train)

    print("training data")
    y_pred = model.predict(X_train)
    print(score_fun(y_train, y_pred))

    print("testing data")
    y_pred = model.predict(X_test)
    print(score_fun(y_test, y_pred))

    return model

def test_classifier(model,X,y, random_state = 0, score = 'f1',plot_partial=False,return_data=False,test_size=0.5):
    score_funs = {
    'f1' : f1_score, 'precision' : precision_score
    }
    score_fun = score_funs[score]
    
    #split the data into training / testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=test_size, stratify = y, random_state= random_state)
    #X_train = normalize(X_train)
    #X_test = normalize(X_test)
    #cross validate with f1 scores
    scores = cross_val_score(model,X,y,scoring = score,cv=3)

    print('cross validation scores:', scores)
    print('mean,std',np.mean(scores),np.std(scores))
    #fit data and compare scores for training and testing data
    model = model.fit(X_train, y_train)
    
    print("training data")
    y_pred = model.predict(X_train)
    print(score_fun(y_train, y_pred))
    print(classification_report(y_train,y_pred))
    print("testing data")
    y_pred = model.predict(X_test)
    print(score_fun(y_test, y_pred))
    print(classification_report(y_test,y_pred))
    #plot confusion matrix and permutation importances
    fig, axes = plt.subplots(1,3,figsize=(18,5))
    (cm_ax,roc_ax,pi_ax) = axes.ravel()
    plot_confusion_matrix(model,X_test,y_test,ax=cm_ax)
    cm_ax.set_title('confusion matrix')
    #plot_roc_curve(model,X_test,y_test,ax=roc_ax)
    #roc_ax.plot([0,1],[0,1],c='red')
    plot_precision_recall_curve(model,X_test,y_test,ax=roc_ax)
    
    
    result = permutation_importance(model, X_test, y_test, n_repeats=40,
                                random_state=random_state, n_jobs=1)
    sorted_idx = result.importances_mean.argsort()

    pi_ax.boxplot(result.importances[sorted_idx].T,
               vert=False, labels=X_test.columns[sorted_idx])
    pi_ax.set_title("Permutation Importances (test set)")
    fig.tight_layout()
    plt.show()
    
    if plot_partial:
        plot_partial_dependence(model,X,np.arange(X.shape[-1]))
    if return_data:
        return model, (X_train, X_test, y_train, y_test)
    else:
        return model #returns fitted estimator

def pred_thresh(model,X,th,label_i = 1):
    probs = model.predict_proba(X)
    return (probs[:,label_i] > th).astype(np.int)

##Adapted from https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5),scoring='r2'):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel(scoring)

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True,scoring=scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel(scoring)
    axes[2].set_title("Performance of the model")

    return plt

