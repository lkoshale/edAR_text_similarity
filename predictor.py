
import os
import numpy as np
from tensorflow import keras
import tensorflow as tf

class MyPredictor(object):
    """An example Predictor for an AI Platform custom prediction routine."""
    model_dir_global=[]
    def __init__(self, model):
        """Stores artifacts for prediction. Only initialized via `from_path`.
        """
        self._model = model
        

    def predict(self, instances, **kwargs):
        """Performs custom prediction.

        Preprocesses inputs, then performs prediction using the trained Keras
        model.

        Args:
            instances: A list of prediction input instances.
            **kwargs: A dictionary of keyword args provided as additional
                fields on the predict request body.

        Returns:
            A list of outputs containing the prediction results.
        """
        inputs = instances
        results=[]
        with tf.Session(graph=tf.Graph()) as sess:
            tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], model_dir_global[0])
            graph = tf.get_default_graph()
            malstm_input = graph.get_tensor_by_name("malstm_input:0")
            X_test_msg1_embed = graph.get_tensor_by_name("X_test_msg1_embed:0")
            X_test_msg2_embed = graph.get_tensor_by_name("X_test_msg2_embed:0")
            results.append(sess.run(X_test_msg1_embed,{malstm_input:['Iron man','Steel man']}))
            results.append(sess.run(X_test_msg2_embed,{malstm_input:['Iron man','Steel man']}))
        
      
        return results

    @classmethod
    def from_path(cls, model_dir):
        """Creates an instance of MyPredictor using the given path.

        This loads artifacts that have been copied from your model directory in
        Cloud Storage. MyPredictor uses them during prediction.

        Args:
            model_dir: The local directory that contains the trained Keras
                model and the pickled preprocessor instance. These are copied
                from the Cloud Storage model directory you provide when you
                deploy a version resource.

        Returns:
            An instance of `MyPredictor`.
        """
        model_dir_global.append(model_dir) 
        with tf.Session(graph=tf.Graph()) as sess:
            model = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], model_dir)
            
        return cls(model)
