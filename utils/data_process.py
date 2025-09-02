# Imports
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import PorterStemmer
import numpy as np 

def process_tweet(positive_tweets, negative_tweets):
    """
    Processes tweet by tokenizing, lowercasing, removing stop words, and stemming
    them.
    Parameters:
     - positive_tweets: List of positive sentiment tweets (list)
     - negative_tweets: List of negative sentiment tweets (list)
    Returns:
     - positive_tweets: List of processed positive sentiment tweets (List)
     - negative_tweets: List of processed negative sentiment tweets (List)
    """
    # Download the stop words
    nltk.download('stopwords')

    # Tokenize both positive and negative sentiment tweets
    for i in range(0, len(positive_tweets)):
        positive_tweets[i] = word_tokenize(positive_tweets[i].lower())
        negative_tweets[i] = word_tokenize(negative_tweets[i].lower())

    # Get the set of stop words
    stop_words = set(stopwords.words('english'))

    # Remove stop words 
    for i in range(0, len(positive_tweets)):
        positive_tweets[i] = [word for word in positive_tweets[i] if word not in stop_words]
        negative_tweets[i] = [word for word in negative_tweets[i] if word not in stop_words]

    # Create an instance of the porter stemmer
    stemmer = PorterStemmer()
    # Apply porter stemming to each word in each tweet
    for i in range(0, len(positive_tweets)):
        positive_tweets[i] = [stemmer.stem(word) for word in positive_tweets[i]]
        negative_tweets[i] = [stemmer.stem(word) for word in negative_tweets[i]]
    

    return positive_tweets, negative_tweets

def find_frequencies(positive_tweets, negative_tweets):
    """
    Finds the frequency of each word in the corpuses.
    Parameters:
     - positive_tweets - The list of positive sentiment tweets (List)
     - negative_tweets - List of negative sentiment tweets (List)
    
    Returns:
     - positive_freq - Frequencies of each word found in the positive sentiment corpus (Dictionary)
     - negative_freq - Frequencies of each word found in the negative sentiment corpus (Dictionary)    
    """

    positive_freq = {}
    negative_freq = {}

    # Find the frequencies of each word in the positive sentiment corpus
    for tweet in positive_tweets:
        for word in tweet:
            if word not in positive_freq.keys():
                positive_freq[word] = 1
            else:
                positive_freq[word] += 1
    
    # Find the frequencies of each word in the negative sentiment corpus
    for tweet in negative_tweets:
        for word in tweet:
            if word not in negative_freq.keys():
                negative_freq[word] = 1
            else:
                negative_freq[word] += 1
    
    return positive_freq, negative_freq

def build_feature_vector(tweet, positive_freq, negative_freq):
    """
    Given a tweet, positive frequencies, negative frequencies, build a feature
    vector of size 3 where the first value is the bias, second value is 
    total number of positive frequencies, and third is total number of negative
    frequencies.
    Parameters:
        - tweet - The tweet being converted to feature vector (List)
        - positive_freq - Frequencies of each word found in the positive sentiment corpus (Dictionary)
        - negative_freq - Frequencies of each word found in the negative sentiment corpus (Dictionary)
    
    Returns:
        - feature_vector - The feature vector of tweet (NumPy Array)
    """
    
    # Create a numpy array of size three
    feature_vector = np.zeros(3)
    word_checked = []
    
    # Set the bias value to one
    feature_vector[0] = 1

    # Iterate through each word in the tweet
    for word in tweet:
        # If word frequency has not been added.
        if word not in word_checked:
            feature_vector[1] += positive_freq.get(word, 0)
            feature_vector[2] += negative_freq.get(word, 0)
            word_checked.append(word)
    
    # Return the feature vector
    return feature_vector 

        
    

