from flask import Flask, render_template, url_for, request


class FakeCovApp(object):
    def __init__(self, host='0.0.0.0', port=1234, debug=True):
        self.host = host
        self.port = port
        self.debug_mode = debug

    def run(self):
        app = Flask(__name__)

        @app.route('/', methods=['POST', 'GET'])
        def index():
            dummy_val = None

            if request.method == 'POST':
                # fetch Tweets
                if request.form['submitBtn'] == 'fetch-tweet':
                    # TODO: get tweets via Tweepy
                    dummy_val = 'FETCH TWEETS'
                # predict written text
                elif request.form['submitBtn'] == 'predict-text':
                    text = request.form.get('ownText')
                    dummy_val = self.predict(text)

            return render_template(
                "index.jinja2",
                dummy_var=dummy_val
            )

        app.run(host=self.host, port=self.port, debug=self.debug_mode)

    def predict(self, txt):
        self._preprocess(txt)
        # TODO: predict preprocessed text
        return f'PREDICT {txt}'

    def _preprocess(self, txt):
        # TODO: preprocess text
        pass


if __name__ == '__main__':
    fake_cov_app = FakeCovApp()
    fake_cov_app.run()
