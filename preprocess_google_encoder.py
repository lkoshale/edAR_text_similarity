import numpy as np
import tensorflow_hub as hub
import tensorflow as tf

class MyGoogleEncoder(object):
  def __init__(self):
    self.random = None

  def preprocess(self, data):
    embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")  
    
    with tf.Session() as session:
      session.run([tf.global_variables_initializer(), tf.tables_initializer()])
      test_embed = session.run(embed(data))
    test_embed = test_embed[:,np.newaxis]
    return test_embed
