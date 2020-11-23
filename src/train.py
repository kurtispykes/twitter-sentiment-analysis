import argparse

import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

import config
import model_dispatcher
import preprocessing as pp


def run_training(model:str) -> None:
    """
    Training our Machine Learning model and serializing to disc
    """
    # read train data
    df = pd.read_csv(config.ORIGINAL_TRAIN)
    # shuffle data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    # create new column "all_text"
    df[config.ALL_TEXT] = df[config.TEXT] + df[config.KEYWORD].fillna("none") + df[config.LOCATION].fillna("none")
    # split into features and labels
    X = df.drop([config.TEXT, config.KEYWORD, config.LOCATION, config.TARGET], axis=1)
    y = df[config.TARGET]
    del df

    # process tweets
    X[config.ALL_TEXT] = X[config.ALL_TEXT].apply(pp.process_tweet)

    # initialize kfold
    skf = StratifiedKFold(n_splits=5, shuffle=False)
    for fold, (train_idx, val_idx) in enumerate(skf.split(X=X, y=y)):
        X_train, X_val = X.loc[train_idx, :], X.loc[val_idx, :]
        y_train, y_val = y[train_idx], y[val_idx]

        # vectorize text and store model
        tfidf_vect = TfidfVectorizer()
        X_train_vect = tfidf_vect.fit_transform(X_train[config.ALL_TEXT].values)
        X_val_vect = tfidf_vect.transform(X_val[config.ALL_TEXT].values)
        joblib.dump(tfidf_vect, f"{config.MODEL_DIR}/tfidf_vec_{model}_{fold}.pkl")

        clf = model_dispatcher.MODELS[model]
        clf.fit(X_train_vect, y_train)
        y_preds = clf.predict(X_val_vect)
        print(f"F1 Score: {metrics.f1_score(y_val, y_preds)}")
        joblib.dump(clf, f"{config.MODEL_DIR}/{model}_tfidf_{fold}.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str
    )
    args = parser.parse_args()
    run_training(model=args.model)


