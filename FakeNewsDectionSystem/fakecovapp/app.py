import argparse
import pickle as pkl
import numpy as np
import json

import pandas as pd
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
        self.tokenizer = self._read_file(self.cfg['tokenizer_path'], 'pkl')
        # Tweepy
        auth = tweepy.OAuthHandler(self.cfg['tweepy']['CONSUMER_KEY'], self.cfg['tweepy']['CONSUMER_SECRET'])
        auth.set_access_token(self.cfg['tweepy']['ACCESS_TOKEN'], self.cfg['tweepy']['ACCESS_TOKEN_SECRET'])
        self.twitter_api = tweepy.API(auth)

        self.data_table = pd.DataFrame()

    def run(self):
        app = Flask(__name__)

        @app.route('/', methods=['POST', 'GET'])
        def index():
            show_table = False

            if request.method == 'POST':
                # fetch Tweets
                if request.form['submitBtn'] == 'fetch-tweet':
                    self.data_table = self.get_tweets(
                        search_string=request.form.get('searchString'),
                        date_string=request.form.get('sinceDate'),
                        tweet_amount=request.form.get('tweetAmount')
                    )
                    show_table = True

                # predict written text
                elif request.form['submitBtn'] == 'predict-text':
                    text = request.form.get('ownText')
                    pred_fake, pred_real = self.predict([text])
                    self.data_table = pd.DataFrame(
                        {
                            'text': [text],
                            'fake': pred_fake,
                            'real': pred_real
                        }
                    )
                    show_table = True

                # predict tweets
                elif request.form['submitBtn'] == 'predict-tweet':
                    pred_fake, pred_real = self.predict(self.data_table['tweet'].values.tolist())
                    self.data_table['fake'] = pred_fake
                    self.data_table['real'] = pred_real
                    show_table = True

            return render_template(
                "index.jinja2",
                show_table=show_table,
                table_columns=self.data_table.columns if show_table else None,
                table_data=self.data_table.values.tolist() if show_table else None,
                predict_tweet_btn=True if 'user' in self.data_table.columns else False
            )

        app.run(host=self.host, port=self.port, debug=self.debug_mode)

    def get_tweets(self, search_string, date_string, tweet_amount):
        """
        Fetch tweets  from twitter via tweepy

        :param search_string: str
        :param date_string: str - date as string in format YYYY-MM-DD
        :param tweet_amount: int
        :return: pd.DataFrame
        """
        user_l = list()
        tweet_l = list()
        for tweet in tweepy.Cursor(self.twitter_api.search_tweets, q=search_string, lang="en",
                               since=date_string, tweet_mode='extended').items(int(tweet_amount)):
            username = tweet.user.screen_name
            try:
                tweet_text = tweet.retweeted_status.full_text
            except AttributeError:
                tweet_text = tweet.full_text

            user_l.append(username)
            tweet_l.append(tweet_text)

        return pd.DataFrame({'user': user_l, 'tweet': tweet_l})

    def predict(self, txt):
        """
        Method to tokenize, pad and predict text
        :param txt: list - contains strings
        :return: tuple - (list for fake, list for real)
        """
        # self._preprocess(txt)
        tokens = self.tokenizer.texts_to_sequences(txt)
        padded = pad_sequences(tokens, maxlen=self.cfg['model']['MAX_SEQUENCE_LENGTH'])
        pred = self.model.predict(padded)
        return [np.round(p[0], decimals=1) for p in pred], [np.round(p[1], decimals=1) for p in pred]

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
