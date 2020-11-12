import logging

import pandas as pd
from sklearn.model_selection import StratifiedKFold

import config

_logger = logging.getLogger(__name__)

def get_folds() -> None:
    """
    Create new train data with column indicating what fold the instance falls into.
    """
    # read training data
    df = pd.read_csv(config.ORIGINAL_TRAIN)
    # populate new "kfold" column
    df[config.FOLD] = -1
    # shuffle data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    # initialize kfold
    skf = StratifiedKFold(n_splits=5, shuffle=False)
    for fold, (train_idx, val_idx) in enumerate(skf.split(X=df, y=df[config.TARGET])):
        _logger.info(f"Fold {fold}/nTrain: {len(train_idx)}/nTest: {len(val_idx)}")
        df.loc[val_idx, config.FOLD] = fold
    # save dataframe
    df.to_csv(config.MODIFIED_TRAIN, index=False)
    _logger.info(f"Data Saved to {config.MODIFIED_TRAIN}")

if __name__ == "__main__":
    get_folds()
