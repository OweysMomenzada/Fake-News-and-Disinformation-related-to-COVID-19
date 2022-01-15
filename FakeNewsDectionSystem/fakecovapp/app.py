import argparse
import pickle as pkl
import numpy as np
import json
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


class FakeCovApp(object):
    def __init__(self, host='0.0.0.0', port=1234, debug=True, cfg_path='fakecov_config.json'):
        self.host = host
        self.port = port
        self.debug_mode = debug
        self.cfg = self._read_file(cfg_path, 'json')
        self.model = self._load_model()
        self.tokenizer = self._read_file(self.cfg['tokenizer_path'], 'pkl')

    def run(self):
        app = Flask(__name__)

        @app.route('/', methods=['POST', 'GET'])
        def index():
            dummy_val = None

            if request.method == 'POST':
                # fetch Tweets
                if request.form['submitBtn'] == 'fetch-tweet':
                    dummy_val = self.get_tweets(
                        search_string=request.form.get('searchString'),
                        date_string=request.form.get('sinceDate'),
                        tweet_amount=request.form.get('tweetAmount')
                    )

                # predict written text
                elif request.form['submitBtn'] == 'predict-text':
                    text = request.form.get('ownText')
                    pred_fake, pred_real = self.predict(text)
                    dummy_val = f'{text}: {pred_fake}(fake) / {pred_real}(real)'

            return render_template(
                "index.jinja2",
                dummy_var=dummy_val
            )

        app.run(host=self.host, port=self.port, debug=self.debug_mode)

    def get_tweets(self, search_string, date_string, tweet_amount):
        # TODO: get tweets via Tweepy
        return f'FETCH {tweet_amount} TWEETS words: \'{search_string}\' / date: \'{date_string}\''

    def predict(self, txt):
        """
        Method to tokenize and predict text
        :param txt: str
        :return: tuple - (val for fake, val for real)
        """
        # self._preprocess(txt)
        tokens = self.tokenizer.texts_to_sequences([txt])
        padded = pad_sequences(tokens, maxlen=self.cfg['model']['MAX_SEQUENCE_LENGTH'])
        array_pred = np.round(self.model.predict(padded)[0], 4)
        return array_pred[0], array_pred[1]

    def _preprocess(self, txt):
        # TODO: preprocess text
        pass

    def _load_model(self):
        model = load_model(self.cfg['model']['model_path'])
        return model

    @staticmethod
    def _read_file(path, file_type):
        with open(path, 'rb') as fh:
            output = eval(file_type).load(fh)
        return output


if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help='Path to config.json file')

    args = parser.parse_args()
    # run app
    fake_cov_app = FakeCovApp(cfg_path=args.config if args.config else 'fakecov_config.json')
    fake_cov_app.run()
