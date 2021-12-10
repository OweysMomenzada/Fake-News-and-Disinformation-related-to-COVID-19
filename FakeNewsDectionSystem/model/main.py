import numpy as np
import pickle as pkl

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, SpatialDropout1D, LSTM


model_path = "model_pretrain/model"
MAX_LEN = 512
MAX_NB_WORDS = 29000
MAX_SEQUENCE_LENGTH = 150
HIDDEN_DIM = 150
TOKENIZER = Tokenizer(num_words=MAX_NB_WORDS, lower=True)
X = np.load('FakeDetection.pkl', allow_pickle=True)


class FakeCov:

    def __new__(self):
        model = self.CovFake_model()
        model.load_weights(model_path)

        return model

    @staticmethod
    def CovFake_model():
        model = Sequential()
        model.add(Embedding(MAX_NB_WORDS, HIDDEN_DIM, input_length=X.shape[1]))
        model.add(SpatialDropout1D(0.2))
        model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(2, activation='softmax'))

        return model


def get_tokens():
    with open('tokenizer.pkl', 'rb') as handle:
        tokenizer_load = pkl.load(handle)

    return tokenizer_load


def fakecov_predict(text,model):
    tokenizer = get_tokens()
    text = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(text, maxlen=MAX_SEQUENCE_LENGTH)
    array_pred = np.round(model.predict(padded)[0], 4)

    pred = {'fake': array_pred[0],
            'real': array_pred[1]
            }

    return pred


if __name__ == "__main__":
    fakecov_inst = FakeCov()
    print(fakecov_predict('The CDC currently reports 99031 deaths. In general the discrepancies in death counts between different sources are small and explicable. The death toll stands at roughly 100000 people today.',fakecov_inst))
