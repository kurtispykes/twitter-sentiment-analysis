import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

nltk.download("stopwords")

def clean_tweet(tweet:str) -> str:
    """
    Clean the hashtags, hyperlinks, and punctuation from tweets and convert all text to lowercase
    :param tweet: tweet by a unique user
    :return: cleaned string without hashtags, emojis, and punctuation
    """
    # make text lower case
    tweet = tweet.lower()
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', str(tweet))
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', str(tweet))
    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', str(tweet))
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', str(tweet))

    # remove stopwords
    stop_words = set(stopwords.words())
    tweet = [word for word in tweet if word not in stop_words]

    # remove punctuation
    tweets_cleaned = []
    for word in tweet:
        if word not in string.punctuation and word:
            tweets_cleaned.append(word)

    return "".join(tweets_cleaned)

