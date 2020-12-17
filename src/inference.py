import os
import argparse

import gensim
import joblib
import pandas as pd

import config
import features as f

def predict(model_type: str,
            test_data:pd.DataFrame=config.MODIFIED_TEST,
            model_path:str=config.MODEL_DIR
            ):
    # read data
    df_test = pd.read_csv(test_data)
    # load train embeddings
    wv_model = gensim.models.Word2Vec.load(f"{config.MODEL_DIR}SKIP_GRAM_{model_type}_/SKIP_GRAM_embeddings")
    X = pd.DataFrame(f.get_embeddings(df_test[config.TOKENS], wv_model))

    predictions = None
    # loop through all folds
    for FOLD in range(5):
        # load the classifier
        clf = joblib.load(os.path.join(model_path, f"SKIP_GRAM_{model_type}_/{model_type}_SKIP_GRAM_{FOLD}.pkl"))
        preds = clf.predict(X)

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
    sample_sub.to_csv(f"{config.MODEL_DIR}SKIP_GRAM_{args.model_type}_/{args.model_type}.csv", index=False)