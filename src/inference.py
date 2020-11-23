import os
import argparse

import joblib
import pandas as pd

import config
import preprocessing as pp

def predict(model_type: str,
            test_data:pd.DataFrame=config.TEST_DATA,
            model_path:str=config.MODEL_DIR
            ):
    # read data
    df = pd.read_csv(test_data)
    # combine all the text fields
    df[config.ALL_TEXT] = df[config.TEXT] + df[config.KEYWORD].fillna("none") + df[config.LOCATION].fillna("none")
    # process the tweets
    df[config.ALL_TEXT] = df[config.ALL_TEXT].apply(pp.process_tweet)
    predictions = None
    # loop through all folds
    for FOLD in range(5):
        # load the vectorizer
        vectorizer = joblib.load(os.path.join(model_path, f"tfidf_vec_{model_type}_{FOLD}.pkl"))
        df_test = vectorizer.transform(df[config.ALL_TEXT].values)
        clf = joblib.load(os.path.join(model_path, f"{model_type}_tfidf_{FOLD}.pkl"))

        preds = clf.predict(df_test)

        if FOLD == 0:
            predictions = preds
        else:
            predictions += preds

    predictions //= 5

    return predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        type=str
    )

    args = parser.parse_args()
    submission = predict(model_type=args.model_type)
    sample_sub = pd.read_csv(config.SUBMISSION)
    sample_sub.loc[:, config.TARGET] = submission
    sample_sub.to_csv(f"{config.MODEL_DIR}{args.model_type}.csv", index=False)