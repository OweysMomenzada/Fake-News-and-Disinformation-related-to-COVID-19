import os

import pandas as pd
import numpy as np
import re

import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel

bert = TFAutoModel.from_pretrained("./bert_model/")
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
model_path = "./model_pretrain/model"
max_length = 512


class FakeCov:

    def __new__(self):
        model = self.CovFake_model()
        model.load_weights(model_path)

        return model

    @staticmethod
    def CovFake_model():
        input_ids = tf.keras.layers.Input(shape=(max_length,), name='input_ids', dtype='int32')
        input_mask = tf.keras.layers.Input(shape=(max_length,), name='attention_mask', dtype='int32')

        embedding = bert(input_ids, attention_mask=input_mask)[0]
        x = tf.keras.layers.GlobalMaxPool1D()(embedding)
        x = tf.keras.layers.GlobalAveragePooling1D()(embedding)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        model = tf.keras.Model(inputs=[input_ids, input_mask], outputs=output)

        model.layers[2].trainable = False

        return model


def tokenize_slice_text(text):
    tokens = tokenizer.encode_plus(text, max_length=max_length, padding='max_length', add_special_tokens=True,
                                   truncation=True, return_token_type_ids=False, return_attention_mask=True,
                                   return_tensors='tf')

    Xids = tokens['input_ids']
    Xmask = tokens['attention_mask']

    return [Xids, Xmask]


def fakecov_predict(text,model):
    text = tokenize_slice_text(text)
    prediction = model.predict(text)

    return prediction


if __name__ == "__main__":
    fakecov_inst = FakeCov()
    print(fakecov_predict('Oil can heal you from corona!',fakecov_inst))

