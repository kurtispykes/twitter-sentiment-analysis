import logging
import argparse

import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import metrics

import config
import model_dispatcher
import preprocessing as pp

_logger = logging.getLogger(__name__)

def run_training(model:str, fold:int) -> None:
    """
    Training our Machine Learning model and serializing to disc
    """
    # read training data
    df = pd.read_csv(config.MODIFIED_TRAIN)
    # split to train and validation fold
    train = df[df[config.FOLD] != fold].reset_index(drop=True)
    valid = df[df[config.FOLD] == fold].reset_index(drop=True)
    # create one text column
    train[config.ALL_TEXT] = train[config.TEXT] + train[config.KEYWORD] + train[config.LOCATION]
    valid[config.ALL_TEXT] = valid[config.TEXT] + valid[config.KEYWORD] + valid[config.LOCATION]
    # drop unnecessary columns
    X_train = train.drop([config.TEXT, config.KEYWORD, config.LOCATION, config.TARGET, config.FOLD], axis=1)
    y_train = train[config.TARGET]
    X_valid = valid.drop([config.TEXT, config.KEYWORD, config.LOCATION, config.TARGET, config.FOLD], axis=1)
    y_valid = valid[config.TARGET]
    # delete df from memory
    del df

    # process tweets
    X_train[config.ALL_TEXT] = X_train[config.ALL_TEXT].apply(pp.process_tweet)
    X_valid[config.ALL_TEXT] = X_valid[config.ALL_TEXT].apply(pp.process_tweet)

    # vectorize train data (only on text)
    count_vec = CountVectorizer()
    train_vecs = count_vec.fit_transform(X_train[config.ALL_TEXT])
    test_vecs = count_vec.transform(X_valid[config.ALL_TEXT])

    joblib.dump(count_vec, f"{config.MODEL_DIR}/count_vec_{model}_{fold}.pkl" )
    _logger.info("Persisted Vectorizer")

    clf = model_dispatcher.MODELS[model]
    clf.fit(train_vecs, y_train)
    preds = clf.predict(test_vecs)
    print(f"F1 Score: {metrics.f1_score(y_valid, preds)}")

    joblib.dump(clf, f"{config.MODEL_DIR}/{model}_{fold}.pkl")
    _logger.info("Persisted classifier")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str
    )
    parser.add_argument(
        "--fold",
        type=int
    )
    args = parser.parse_args()
    run_training(model=args.model, fold=args.fold)


