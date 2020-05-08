from collections import Counter, namedtuple
from nltk.tree import Tree
import numpy as np
import os
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
import scipy.stats
import utils

__author__ = "Adrian Sarno, Jennifer Arnold"
__version__ = "CS224u, Stanford, Spring 2020"


def slots_domain_intent_class_func(y):
    """Define a combined slot-domain-intent classification task.

    Parameters
    ----------
    y : str
        Assumed to be an ASIC label string.

    Returns
    -------
    str or None
        None values are ignored by `build_dataset` and thus left out of
        the experiments.

    """
    y = y.split()
    if len(y) > 1:
        slot_labels, intent_label = y[:-1], y[-1]
        return [slot_labels, "airline_travel", intent_label]
    else:
        return None


def airline_travel_flat_reader(src_filename, class_func=slots_domain_intent_class_func):
    """Iterator for the flat format of the Airline Travel 
    Information System dataset. The iterator yields (sentence, label) pairs.

    The labels are tuples consisting of a list of IOB-tagged slot names,
    followed by the domain and intent names

    Parameters
    ----------
    src_filename : str
        Full path to the file to be read e.g.: atis-2.dev.w-intent.iob
    class_func : function mapping labels to labels.
        If this is not defined, then the default function will return 
        IOB-tagged slot names, followed by the domain and intent labels.
        Other options: `intent_class_func` and `domain_class_func`
        (or you could write your own).

    Yields
    ------
    (sentence, label)
        nltk.Tree, str in {'0','1','2','3','4'}
    
    """
    # Example
    # BOS i want to fly from boston at 838 am and arrive in denver at 1110 in the morning EOS	O O O O O O B-fromloc.city_name O B-depart_time.time I-depart_time.time O O O B-toloc.city_name O B-arrive_time.time O O B-arrive_time.period_of_day atis_flight
    with open(src_filename, encoding='utf8') as f:
        for line in f:
            sentence, label = line.split("EOS")  # strip out the EOS token, because it has no label
            sentence = sentence[3:].strip()      # remove BOS
            label = label.strip()[2:]            # remove BOS tag
            label = class_func(label)
            # If the example doesn't fall into any of the classes 
            # for this version of the problem, then we drop 
            # the example:
            if label:
                yield (sentence, label)

def slot_filling_and_intent_func(y):
    
    y = slots_domain_intent_class_func(y)
    if len(y) == 3:
        return (y[0], y[2])
    else:
        return None

def slot_filling_func(y):
    """Define slot filling (sequence labeling) task.

    Parameters
    ----------
    y : str
        Assumed to be an ASIC label string.

    Returns
    -------
    str or None
        None values are ignored by `build_dataset` and thus left out of
        the experiments.

    """
    y = slots_domain_intent_class_func(y)
    if len(y) == 3:
        return y[0]
    else:
        return None


def domain_class_func(y):
    """Define an domain classification task.

    Parameters
    ----------
    y : str
        Assumed to be an ASIC label string.

    Returns
    -------
    str or None
        None values are ignored by `build_dataset` and thus left out of
        the experiments.

    """
    y = slots_domain_intent_class_func(y)
    if len(y) == 3:
        return y[1]
    else:
        return None

    
def intent_class_func(y):
    """Define an intent classification task.

    Parameters
    ----------
    y : str
        Assumed to be an ASIC label string.

    Returns
    -------
    str or None
        None values are ignored by `build_dataset` and thus left out of
        the experiments.

    """
    y = slots_domain_intent_class_func(y)
    if len(y) == 3:
        return y[2]
    else:
        return None

    
def train_reader(atis_home, **kwargs):
    """Convenience function for reading the train file, flat format with intent."""
    src = os.path.join(atis_home, 'atis-2.train.w-intent.iob')
    return airline_travel_flat_reader(src, **kwargs)


def dev_reader(atis_home, **kwargs):
    """Convenience function for reading the dev file, flat format with intent."""
    src = os.path.join(atis_home, 'atis-2.dev.w-intent.iob')
    return airline_travel_flat_reader(src, **kwargs)


def test_reader(atis_home, **kwargs):
    """Convenience function for reading the test file, flat format with intent.
    This function should be used only for the final stages of a project,
    to obtain final results.
    """
    src = os.path.join(atis_home, 'atis.test.w-intent.iob')
    return airline_travel_flat_reader(src, **kwargs)


def build_dataset(atis_home, 
                  reader, 
                  phi, 
                  batch_phi, 
                  label_alignment_func, 
                  class_func, 
                  vectorizer=None, 
                  vectorize=True):
    """Core general function for building experimental datasets.
    Preprocessing of the data.

    Parameters
    ----------
    atis_home : str
        Full path to the 'ATIS' dataset directory.
    reader : iterator
       Should be `train_reader`, `dev_reader`, or another function
       defined in those terms. This is the dataset we'll be
       featurizing.
    phi : feature function
       Any function that takes an ATIS sentence as input
       and returns a feature representation as output.
       The output can be a bool/int/float-valued dict (BOW featurizers)
       or a list of sub-word embeddings (BERT featurization)
    batch_phi : batch featurization function
        Any function that takes a list of ATIS sentences as input
        and returns a 3-D numpy array of token-level embeddings as output.
    label_alignment_func : a function that aligns the IOB labels
        to the sub-word tokens used by the featurization function
    class_func : function on the ATIS labels
       Any function like `intent_class_func`.
       This modifies the ATIS labels based on the experimental
       design. If `class_func` returns None for a label, then that
       item is ignored.
    vectorizer : sklearn.feature_extraction.DictVectorizer
       If this is None, then a new `DictVectorizer` is created and
       used to turn the list of dicts created by `phi` into a
       feature matrix. This happens when we are training.
       If this is not None, then it's assumed to be a `DictVectorizer`
       and used to transform the list of dicts. This happens in
       assessment, when we take in new instances and need to
       featurize them as we did in training.
    vectorize : bool
       Whether to use a DictVectorizer. Set this to False for
       deep learning models that process their own input.

    Returns
    -------
    dict
        A dict with keys 'X' (the feature matrix), 'y' (the list of
        labels, aligned but not lenght-padded), 'vectorizer' (the `DictVectorizer`), and
        'raw_examples' (the ATIS examples, for error analysis).

    """
    # read the raw examples and labels.
    # if a single-sentence featurizer (phi) is defined, 
    # call it to compute feature dictionaries (BOW)
    labels = []
    feat_dicts = []
    raw_examples = []
    for sentence, label in reader(atis_home, class_func=class_func):
        labels.append(label)
        if phi != None:
            feat_dicts.append(phi(sentence))
        raw_examples.append(sentence)
    
    # compute the feature matrix:
    # if vectorizing is required (vectorize)
    # call the dictionary vectorizer
    # Optionally, if an existing vectorizer was passed
    # call this instead.
    # (this is used for vectorizing dev a test sets with
    # the vocabulary collected during train)
    # None of this is used for pretrained transformers
    feat_matrix = None
    if vectorize:
        # In training, we want a new vectorizer:
        if vectorizer == None:
            vectorizer = DictVectorizer(sparse=False)
            feat_matrix = vectorizer.fit_transform(feat_dicts)
        # In assessment, we featurize using the existing vectorizer:
        else:
            feat_matrix = vectorizer.transform(feat_dicts)
    else:
        feat_matrix = feat_dicts
        
    # For pretrained transformers, call the batch_phi, that 
    # returns a padded-length feature matrix with an attention_mask
    if batch_phi != None:
        feat_matrix = batch_phi(raw_examples)
  
    # for sequence-tagging tasks with sub-word tokenization,
    # align the labels to the first sub-word of each word
    label_alignment_map = []
    if label_alignment_func != None:
        labels, _, _ = label_alignment_func(raw_examples, labels)
        
    
    return {'X': feat_matrix,
            'y': labels,  # aligned but not lenght-padded
            'vectorizer': vectorizer,
            'raw_examples': raw_examples
           }


def experiment(
        atis_home,
        phi=None,
        batch_phi=None,
        label_alignment_func=None,  # remove?
        train_func=None,
        train_reader=train_reader,
        assess_reader=None,
        train_size=0.7,
        class_func=slots_domain_intent_class_func,
        score_func=utils.safe_macro_f1,
        metrics_report_func=classification_report,
        vectorize=True,
        verbose=True,
        random_state=None):
    """Generic experimental framework for ATIS. Either assesses with a
    random train/test split of `train_reader` or with `assess_reader` if
    it is given.

    Parameters
    ----------
    atis_home : str
        Full path to the 'ATIS' dataset directory.
    phi : feature function
       Any function that takes an ATIS sentence as input
       and returns a feature representation as output.
       The output can be a bool/int/float-valued dict (BOW featurizers)
       or a list of sub-word embeddings (BERT featurization)
    batch_phi : batch featurization function
        Any function that takes a list of ATIS sentences as input
        and returns a 3-D numpy array of token-level embeddings as output.
    label_alignment_func : a function that aligns the IOB labels
        to the sub-word tokens used by the featurization function
    train_func : model wrapper
        Any function that takes a feature matrix and a label list
        as its values and returns a fitted model with a `predict`
        function that operates on feature matrices.
    train_reader : ATIS iterator (default: `train_reader`)
        Iterator for training data.
    assess_reader : iterator or None (default: None)
        If None, then the data from `train_reader` are split into
        a random train/test split, with the the train percentage
        determined by `train_size`. If not None, then this should
        be an iterator for assessment data (e.g., `dev_reader`).
    train_size : float (default: 0.7)
        If `assess_reader` is None, then this is the percentage of
        `train_reader` devoted to training. If `assess_reader` is
        not None, then this value is ignored.
    class_func : function on the ATIS labels
        Any function like `intent_class_func`.
        This modifies the ATIS labels based on the experimental
        design. If `class_func` returns None for a label, then that
        item is ignored.
    score_metric : function name (default: `utils.safe_macro_f1`)
        This should be an `sklearn.metrics` scoring function. The
        default is weighted average F1 (macro-averaged F1). For
        comparison with the SST literature, `accuracy_score` might
        be used instead. For micro-averaged F1, use
          (lambda y, y_pred : f1_score(y, y_pred, average='micro', pos_label=None))
        For other metrics that can be used here, see
        see http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
    vectorize : bool
        Whether to use a DictVectorizer for creating the dataset. 
        If the flag vectorize is False, then the experiment will expect that phi returns
        vectors that can be passed directly to the model created by the train_func.
        In the case of pretrained transformer models, the BERT hidden states returned by batch_phi 
        are passed directly to the neural netowork classification model returned by train_func
    verbose : bool (default: True)
        Whether to print out the model assessment to standard output.
        Set to False for statistical testing via repeated runs.
    random_state : int or None
        Optionally set the random seed for consistent sampling.

    Prints
    -------
    To standard output, if `verbose=True`
        Model precision/recall/F1 report. Accuracy is micro-F1 and is
        reported because many SST papers report that figure, but macro
        precision/recall/F1 is better given the class imbalances and the
        fact that performance across the classes can be highly variable.

    Returns
    -------
    dict with keys
        'model': trained model
        'phi': the function used for featurization
        'train_dataset': a dataset as returned by `build_dataset`
        'assess_dataset': a dataset as returned by `build_dataset`
        'predict_output': the output of the model.predict() metiond, 
             it contains the predictions on the assessment data, and maybe more.
             the score_func should know how to parse it.
        'metric': `score_func.__name__`
        'score': the `score_func` score on the assessment data

    """
    # Train dataset:
    train = build_dataset(
        atis_home,
        train_reader,
        phi,
        batch_phi, 
        label_alignment_func,
        class_func,
        vectorizer=None,
        vectorize=vectorize)
    
    X_train = train['X']
    y_train = train['y']
    raw_train = train['raw_examples']
    
    # Manage the assessment set-up:
    if assess_reader == None:
        # TODO: include attention mask
        X_train, X_assess, y_train, y_assess, raw_train, raw_assess =\
        train_test_split(X_train, y_train, raw_train,
            train_size=train_size, test_size=None, random_state=random_state)
        assess = {
            'X': X_assess,
            'y': y_assess,
            'vectorizer': train['vectorizer'],
            'raw_examples': raw_assess}
    else:
        # Assessment dataset using the training vectorizer:
        assess = build_dataset(
            atis_home,
            assess_reader,
            phi,
            batch_phi,
            label_alignment_func,
            class_func,
            vectorizer=train['vectorizer'],  # use the train vocabulary
            vectorize=vectorize)
        X_assess, y_assess = assess['X'], assess['y']
        
    # Train:
    mod = train_func(X_train, y_train)
    
    # Predictions: predict output can be a tuple (the metrics_report_func must know what its assiciated model returns
    predict_output = mod.predict(X_assess)
        
    # Compute metrics:
    if verbose:
        print(metrics_report_func(y_assess, predict_output, digits=3))

    score_result = score_func(y_assess, predict_output)

        
    # Return the overall score and experimental info:
    return {
        'model': mod,
        'phi': phi,
        'train_dataset': train,
        'assess_dataset': assess,
        'predictions': predict_output,
        'metric': score_func.__name__,
        'score': score_result}


def compare_models(
        atis_home,
        phi1,
        train_func1,
        phi2=None,
        train_func2=None,
        vectorize1=True,
        vectorize2=True,
        stats_test=scipy.stats.wilcoxon,
        trials=10,
        reader=train_reader,
        train_size=0.7,
        class_func=slots_domain_intent_class_func,
        score_func=utils.safe_macro_f1):
    """Wrapper for comparing models. The parameters are like those of
    `experiment`, with the same defaults, except

    Parameters
    ----------
    atis_home : str
        Full path to the ATIS dataset directory.
    phi1, phi2
        Featurizers, features that preprocess input, converting
        it into numbers.
        Just like `phi` for `experiment`. `phi1` defaults to
        `unigrams_phi`. If `phi2` is None, then it is set equal
        to `phi1`.
    train_func1, train_func2
        Just like `train_func` for `experiment`. If `train_func2`
        is None, then it is set equal to `train_func`.
    vectorize1, vectorize1 : bool
        Whether to vectorize the respective inputs. Use `False` for
        deep learning models that featurize their own input.
    stats_test : scipy.stats function
        Defaults to `scipy.stats.wilcoxon`, a non-parametric version
        of the paired t-test.
    trials : int (default: 10)
        Number of runs on random train/test splits of `reader`,
        with `train_size` controlling the amount of training data.
    train_size : float
        Percentage of data o use for training.
    class_func : function on the ATIS labels
        Any function like `intent_class_func`.
        This modifies the ATIS labels based on the experimental
        design. If `class_func` returns None for a label, then that
        item is ignored.

    Prints
    ------
    To standard output
        A report of the assessment.

    Returns
    -------
    (np.array, np.array, float)
        The first two are the scores from each model (length `trials`),
        and the third is the p-value returned by stats_test.

    """
    if phi2 == None:
        phi2 = phi1
    if train_func2 == None:
        train_func2 = train_func1
    experiments1 = [experiment(atis_home,
        train_reader=reader,
        phi=phi1,
        train_func=train_func1,
        class_func=class_func,
        score_func=score_func,
        vectorize=vectorize1,
        verbose=False) for _ in range(trials)]
    experiments2 = [experiment(atis_home,
        train_reader=reader,
        phi=phi2,
        train_func=train_func2,
        class_func=class_func,
        score_func=score_func,
        vectorize=vectorize2,
        verbose=False) for _ in range(trials)]
    scores1 = np.array([d['score'] for d in experiments1])
    scores2 = np.array([d['score'] for d in experiments2])
    # stats_test returns (test_statistic, p-value). We keep just the p-value:
    pval = stats_test(scores1, scores2)[1]
    # Report:
    print('Model 1 mean: %0.03f' % scores1.mean())
    print('Model 2 mean: %0.03f' % scores2.mean())
    print('p = %0.03f' % pval if pval >= 0.001 else 'p < 0.001')
    # Return the scores for later analysis, and the p value:
    return (scores1, scores2, pval)


def build_rnn_dataset(atis_home, reader, class_func=slots_domain_intent_class_func):
    """Given an ATIS reader, return the `class_func` version of the
    dataset as  (X, y) training pair.

    Parameters
    ----------
    atis_home : str
        Full path to the ATIS data directory.
    reader : train_reader or dev_reader
    class_func : function on the ATIS labels

    Returns
    -------
    X, y
       Where X is a list of list of str, and y is the output label list.

    """
    r = reader(atis_home, class_func=class_func)
    data = [(sentence, label) for sentence, label in r]
    X, y = zip(*data)
    return list(X), list(y)


def build_sentence_dataset(atis_home, reader, class_func=slots_domain_intent_class_func):
    """Given an ATIS reader, return the `class_func` version of the
    dataset. The label of each sentence is set to
    the class for that sentence. We also return the label vector for
    assessment.

    Parameters
    ----------
    atis_home : str
        Full path to the ATIS dataset directory.
    reader : train_reader or dev_reader
    class_func : function on the ATIS labels

    Returns
    -------
    X, y
        Where X is a list of input sentences, and y is the output
        label list.

    """
    data = []
    labels = []
    for (sentence, label) in reader(atis_home, class_func=class_func):
        data.append(sentence)
        labels.append(label)
    return data, labels
