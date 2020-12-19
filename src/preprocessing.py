import re
import string
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

import config

nltk.download("stopwords")

def process_tweet(tweet):
    """
    Process tweet function.
    Input:
        tweet: a string containing a tweet
    Returns:
        tweets_clean: a list of words containing the processed tweet

    *Taken from Coursera NLP Specialization Course 1, week 1 programming
    assignment*
    """
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', str(tweet))
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', str(tweet))
    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', str(tweet))
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', str(tweet))
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if word not in string.punctuation:  # remove punctuation
            tweets_clean.append(word)

    return " ".join(tweets_clean)

def relabel_target(df:pd.DataFrame) -> pd.DataFrame:
    """
    Relabel duplicate tweets that are mislabelled in the training dataset
    :param df: A pandas dataframe with a "target" column
    :return: df
    """
    # copy old target label
    df[config.RELABELED_TARGET] = df[config.TARGET].copy()
    # relabel samples with different labels to their duplicates
    df.loc[df[config.TEXT] == 'like for the music video I want some real action shit like burning buildings and police chases not some weak ben winston shit',
    config.RELABELED_TARGET] = 0
    df.loc[df[config.TEXT] == 'Hellfire is surrounded by desires so be careful and donÛªt let your desires control you! #Afterlife',
    config.RELABELED_TARGET] = 0
    df.loc[df[config.TEXT] == 'To fight bioterrorism sir.',
    config.RELABELED_TARGET] = 0
    df.loc[df[config.TEXT] == '.POTUS #StrategicPatience is a strategy for #Genocide; refugees; IDP Internally displaced people; horror; etc. https://t.co/rqWuoy1fm4',
    config.RELABELED_TARGET] = 1
    df.loc[df[config.TEXT] == 'CLEARED:incident with injury:I-495  inner loop Exit 31 - MD 97/Georgia Ave Silver Spring',
    config.RELABELED_TARGET] = 1
    df.loc[df[config.TEXT] == '#foodscare #offers2go #NestleIndia slips into loss after #Magginoodle #ban unsafe and hazardous for #humanconsumption',
    config.RELABELED_TARGET] = 0
    df.loc[df[config.TEXT] == 'In #islam saving a person is equal in reward to saving all humans! Islam is the opposite of terrorism!',
    config.RELABELED_TARGET] = 0
    df.loc[df[config.TEXT] == 'Who is bringing the tornadoes and floods. Who is bringing the climate change. God is after America He is plaguing her\n \n#FARRAKHAN #QUOTE',
    config.RELABELED_TARGET] = 1
    df.loc[df[config.TEXT] == 'RT NotExplained: The only known image of infamous hijacker D.B. Cooper. http://t.co/JlzK2HdeTG',
    config.RELABELED_TARGET] = 1
    df.loc[df[config.TEXT] == "Mmmmmm I'm burning.... I'm burning buildings I'm building.... Oooooohhhh oooh ooh...",
    config.RELABELED_TARGET] = 0
    df.loc[df[config.TEXT] == "wowo--=== 12000 Nigerian refugees repatriated from Cameroon",
    config.RELABELED_TARGET] = 0
    df.loc[df[config.TEXT] == "He came to a land which was engulfed in tribal war and turned it into a land of peace i.e. Madinah. #ProphetMuhammad #islam",
    config.RELABELED_TARGET] = 0
    df.loc[df[config.TEXT] == "Hellfire! We donÛªt even want to think about it or mention it so letÛªs not do anything that leads to it #islam!",
    config.RELABELED_TARGET] = 0
    df.loc[df[config.TEXT] == "The Prophet (peace be upon him) said 'Save yourself from Hellfire even if it is by giving half a date in charity.'",
    config.RELABELED_TARGET] = 0
    df.loc[df[config.TEXT] == "Caution: breathing may be hazardous to your health.",
    config.RELABELED_TARGET] = 1
    df.loc[df[config.TEXT] == "I Pledge Allegiance To The P.O.P.E. And The Burning Buildings of Epic City. ??????",
    config.RELABELED_TARGET] = 0
    df.loc[df[config.TEXT] == "#Allah describes piling up #wealth thinking it would last #forever as the description of the people of #Hellfire in Surah Humaza. #Reflect",
    config.RELABELED_TARGET] = 0
    df.loc[df[config.TEXT] == "that horrible sinking feeling when youÛªve been at home on your phone for a while and you realise its been on 3G this whole time",
    config.RELABELED_TARGET] = 0
    return df



