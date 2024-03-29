import numpy as np
import re

class MyEncoder(object):
  def __init__(self,vocabulary,inverse_vocabulary,word2vec,stops):
    self.vocabulary = vocabulary
    self.inverse_vocabulary = inverse_vocabulary
    self.word2vec = word2vec
    self.stops = stops
    
  def text_to_word_list(self, text):
    ''' Pre process and convert texts to a list of words '''
    text = str(text)
    text = text.lower()

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    text = text.split() 
    return text  
  
  def preprocess(self, data):
    
    q2n = []  # q2n -> question numbers representation

    for word in data:

      # Check for unwanted words
      if word in self.stops and word not in self.word2vec.vocab:
        continue

      if word not in self.vocabulary:
        self.vocabulary[word] = len(self.inverse_vocabulary)
        q2n.append(len(self.inverse_vocabulary))
        self.inverse_vocabulary.append(word)
      else:
        q2n.append(self.vocabulary[word])

    data = q2n
    data = np.asarray(data)
    data = [data]
    data = np.asarray(data)


    return (data)
