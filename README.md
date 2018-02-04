## Requirements

- python3 ( python2 may require slight tweaking )

## Installation ( These are linux commands, adapt to your machine )

- `$ git clone https://github.com/moejay/gdg-session2.git && cd gdg-session2`
- `$ python3 -m virtualenv venv`
- `$ source venv/bin/activate`
- `(venv)$ pip3 install -r requirements.txt`

## Running the training

- modify trainer.py ( fill in the blanks )
- `(venv)$ python3 trainer.py`

### Optionally run tensorboard

- `(venv)$ tensorboard --logdir=tensorboard`
- Navigate to the address displayed in the console ( use localhost if it doesn't work )

## Running the predictions

- `(venv)$ jupyter notebook`
- The notebook is pretty straightforward, and no modification is needed to run all the way to prediction
- Open Pre-Trained LSTM.ipynb


## Twitter analysis

- head to (twitter apps)[http://apps.twitter.com] ( Signup if you haven't done so already
- Create a new app
- navigate to `Keys and Access Tokens` tab
- Copy consumer key/secret and access token/secret (Might need to be generated) to the auth cell in the notebook


###

This has been adapted from

https://github.com/adeshpande3/LSTM-Sentiment-Analysis