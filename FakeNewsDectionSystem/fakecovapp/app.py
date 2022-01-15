import argparse
import pickle as pkl
import numpy as np
import json
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tweepy


class FakeCovApp(object):
    def __init__(self, host='0.0.0.0', port=1234, debug=True, cfg_path='fakecov_config.json'):
        self.host = host
        self.port = port
        self.debug_mode = debug
        self.cfg = self._read_file(cfg_path, 'json')
        self.model = self._load_model()
        self.tokenizer = self._read_file
        # Tweepy
        auth = tweepy.OAuthHandler(self.cfg['tweepy']['CONSUMER_KEY'], self.cfg['tweepy']['CONSUMER_SECRET'])
        auth.set_access_token(self.cfg['tweepy']['ACCESS_TOKEN'], self.cfg['tweepy']['ACCESS_TOKEN_SECRET'])
        self.twitter_api = tweepy.API(auth)


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
        return_value = ''
        for tweet in tweepy.Cursor(self.twitter_api.search_tweets, q=search_string, lang="en",
                               since=date_string, tweet_mode='extended').items(int(tweet_amount)):
            username = tweet.user.screen_name
            try:
                tweet_text = tweet.retweeted_status.full_text
            except AttributeError:
                tweet_text = tweet.full_text

            return_value += f'Username: \'{username}\' Tweet: \'{tweet_text}\' \n'

        return return_value


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
