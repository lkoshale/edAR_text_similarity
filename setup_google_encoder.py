from setuptools import setup

setup(
    name="my_LSTM_code",
    version='0.2',
    install_requires = ['tensorflow-hub','seaborn','h5py','matplotlib','pandas','pickle'],
    scripts=["predictor_google_encoder.py","preprocess_google_encoder.py"])
