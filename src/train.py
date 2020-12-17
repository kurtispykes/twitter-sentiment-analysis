import os
import argparse

import joblib
import gensim
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

import config
import model_dispatcher
import preprocessing as pp
import features as f

nltk.download("punkt")

def run_training(model:str) -> None:
    """
    Training our Machine Learning model and serializing to disc
    """
    # read train and test data
    df_train = pd.read_csv(config.ORIGINAL_TRAIN)
    df_test = pd.read_csv(config.TEST_DATA)

    # relabel mislabeled samples
    df_train = pp.relabel_target(df_train)

    # shuffle data
    df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)

    # join text on train and test data
    df_train[config.ALL_TEXT] = df_train[config.TEXT]
    df_test[config.ALL_TEXT] = df_test[config.TEXT]

    # clean the newly joined full text
    df_train[config.CLEANED_TEXT] = df_train[config.ALL_TEXT].apply(lambda x: pp.process_tweet(x))
    df_test[config.CLEANED_TEXT] = df_test[config.ALL_TEXT].apply(lambda x: pp.process_tweet(x))

    # create tokens
    df_train[config.TOKENS] = df_train[config.CLEANED_TEXT].apply(lambda x: word_tokenize(x))
    df_test[config.TOKENS] = df_test[config.CLEANED_TEXT].apply(lambda x: word_tokenize(x))

    # create a corpus to train embeddings
    corpus = list(f.fn_pre_process_data(df_train[config.CLEANED_TEXT]))
    corpus += list(f.fn_pre_process_data(df_test[config.CLEANED_TEXT]))

    # train Word2vec (CBOW)
    wv_model = gensim.models.Word2Vec(sentences=corpus, size=150, window=3, min_count=2, sg=1)
    wv_model.train(corpus, total_examples=len(corpus), epochs=10)
    path = os.path.join(config.MODEL_DIR, f"SKIP_GRAM_{model}_")
    os.makedirs(path, exist_ok=True)
    wv_model.save(f"{config.MODEL_DIR}/SKIP_GRAM_{model}_/SKIP_GRAM_embeddings")

    # save the modified train and test data
    df_train.to_csv(config.MODIFIED_TRAIN, index=False)
    df_test.to_csv(config.MODIFIED_TEST, index=False)

    # get the word embeddings
    X = pd.DataFrame(f.get_embeddings(df_train[config.TOKENS], wv_model))
    y = df_train[config.RELABELED_TARGET].copy()

    # initialize kfold
    skf = StratifiedKFold(n_splits=5, shuffle=False)
    for fold, (train_idx, val_idx) in enumerate(skf.split(X=X, y=y)):
        X_train, X_val = X.loc[train_idx, :], X.loc[val_idx, :]
        y_train, y_val = y.loc[train_idx], y.loc[val_idx]

        clf = model_dispatcher.MODELS[model]
        clf.fit(X_train, y_train)
        y_preds_train = clf.predict(X_train)
        y_preds_val = clf.predict(X_val)
        print(f"Fold {fold}")
        print(f"Train f1: {metrics.f1_score(y_train, y_preds_train)}\n"
              f"Val {fold}: {metrics.f1_score(y_val, y_preds_val)}\n")
        joblib.dump(clf, f"{config.MODEL_DIR}/SKIP_GRAM_{model}_/{model}_SKIP_GRAM_{fold}.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str
    )
    args = parser.parse_args()
    run_training(model=args.model)


