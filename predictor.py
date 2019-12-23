
import os
import numpy as np
import tensorflow as tf
import pickle

class MyPredictor(object):
    def __init__(self, model,preprocessor):
        self._model = model
        self._preprocessor = preprocessor

    def predict(self, instances, **kwargs):
        inputs = instances
        preprocessed_inputs_1 = self._preprocessor.preprocess(inputs[0])
        preprocessed_inputs_2 = self._preprocessor.preprocess(inputs[1])
        outputs = self._model.predict([preprocessed_inputs_1,preprocessed_inputs_2])
        return outputs

    @classmethod
    def from_path(cls, model_dir):
        model_path = os.path.join(model_dir, 'malstm_saved_model')
        model = tf.keras.models.load_model(model_path)

        preprocessor_path = os.path.join(model_dir, 'preprocessor.pkl')
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)

        return cls(model, preprocessor)
