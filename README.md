#Disaster Tweet Prediction 
Twitter has become an important communication channel in times of emergency.
The ubiquitousness of smartphones enables people to announce an emergency they’re observing
in real-time. Because of this, more agencies are interested in programmatically monitoring Twitter
(i.e. disaster relief organizations and news agencies). Therefore, in this task I am prediction
whether a given tweet is about a real disaster or not. If so, predict a 1. If not, predict a 0.

##Installation 
###Downloading the Data
- Clone this repository to your computer 
- Navigate to the project directory `cd twitter-sentiment-analysis` from your terminal 
- run `mkdir inputs`
- use `cd inputs` to go into the directory where data should be stored
- Download the data files from Kaggle
    - Data can be found [here](https://www.kaggle.com/c/nlp-getting-started/data)
    - If you don't have a Kaggle account you'd have to create one
    
###Installing the requirements
- Install the requirements using `pip install -r requirements` 
    - The python version is Python 3.8
    - You're better off using virtual environment 

##Usage 

- Navigate to the `src` directory using `cd src` in the project folder
    - Then run `python train.py`
    - This will train an LSTM and create a directory with the `models` directory called `PRETRAIN_WORD2VEC_LSTM` with
    the serialized LSTM and tokenizer inside it. 
    
##Extending This Work
Some ideas to extend this work: 
- Use Different word embeddings
- Try LSTM with attention (See [Attention in Long Short-Term Memory Recurrent Neural Networks](https://machinelearningmastery.com/attention-long-short-term-memory-recurrent-neural-networks/))
- Use a transformer model
- Correct misspelled words  