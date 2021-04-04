import streamlit as st
import zipfile
import tempfile
import time

import random
import os
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras


def load_data():
    input_data = st.sidebar.file_uploader("Upload the data file")
    if input_data is not None:
        data = np.loadtxt(input_data)

        dim_1 = np.shape(data)[0]
        dim_2 = np.shape(data)[1]
        dim_L = (int)(np.sqrt(dim_2))

        data = np.reshape(data, (dim_1, dim_L, dim_L))
        data = np.expand_dims(data, axis=3)

        return data


def load_model():
    input_model = st.sidebar.file_uploader(
        "Upload the trained model", type='zip')

    if input_model is not None:
        zipmodel = zipfile.ZipFile(input_model)
        with tempfile.TemporaryDirectory() as tmp_dir:
            zipmodel.extractall(tmp_dir)
            root_folder = zipmodel.namelist()[0]
            model_dir = os.path.join(tmp_dir, root_folder)
            ml_model = keras.models.load_model(model_dir)

            return ml_model


if __name__ == "__main__":

    data = load_data()
    model = load_model()

    if all(v is not None for v in [data, model]):
        dim_1 = np.shape(data)[0]

        result = model.predict(data)

        img_id = []
        for i in range(9):
            img_id.append(random.randint(0, dim_1))

        df = pd.DataFrame(
            result, columns=['Ferromagnet', 'Antiferromagnet', 'Stripe', 'Paramagnet'])

        phase_i = []
        for j in range(np.shape(result)[0]):
            phase_i.append(np.argmax(result[j]))
        res_count = collections.Counter(np.array(phase_i))

        pred = {
            0: 'Ferromagnet',
            1: 'Antiferromagnet',
            2: 'Stripe',
            3: 'Paramagnet'
        }
        phase_count = {}
        for keys, values in res_count.items():
            # print(pred[keys], values)
            phase_count[pred[keys]] = values

        if st.button("Press to see predictions"):
            st.title("Predictions")
            st.write(df)
            # with st.beta_expander("Press to see counts"):

            st.title("Count of configurations")
            st.write(pd.Series(phase_count, name="Count"))

            st.title("Sample of images uploaded for the prediction")

            fig = plt.figure(figsize=(3, 2))
            for i, j in enumerate(img_id):
                ax = fig.add_subplot(3, 3, (i+1))
                plt.imshow(data[j])
                plt.xticks([])
                plt.yticks([])
            plt.subplots_adjust(left=0.37, right=0.9, wspace=0.01, hspace=0.05)
            st.pyplot(fig)
