import os
import random
import re
import string
import time
from config import settings
ACCESS_TOKEN_FILE=settings['access_token_file']
APP_ID_FILE=settings['app_id_file']
APP_SECRET_FILE=settings['app_secret_file']
LOGGING_DIR=settings['logging_dir']

def getAccessToken():
    if not os.path.exists(ACCESS_TOKEN_FILE):
        return raw_input('Enter access token here: ').strip()
    with open(ACCESS_TOKEN_FILE,'r') as fh:
        return fh.read().strip()

def getAppID():
    if not os.path.exists(APP_ID_FILE):
        return raw_input('Enter app ID here: ').strip()
    with open(APP_ID_FILE,'r') as fh:
        return fh.read().strip()

def getAppSecret():
    if not os.path.exists(APP_SECRET_FILE):
        return raw_input('Enter app secret here: ').strip()
    with open(APP_SECRET_FILE,'r') as fh:
        return fh.read().strip()

def create_long_term_facebook_access_token():
    auth={'SHORT_TERM_ACCESS_TOKEN': getAccessToken(),
          'CLIENT_ID': getAppID(),
          'CLIENT_SECRET': getAppSecret()
         }
    url='https://graph.facebook.com/oauth/access_token?grant_type=fb_exchange_token&client_id=%(CLIENT_ID)s&client_secret=%(CLIENT_SECRET)s&fb_exchange_token=%(SHORT_TERM_ACCESS_TOKEN)s'%auth
    cmd='curl -X GET \"{}\"'.format(url)
    print cmd
    with os.popen(cmd) as fh:
        token=fh.read().strip()
    return token

def startLog(log=True, id_=None):
    import logging
    kwargs_=dict(format='%(asctime)s : %(levelname)s : %(message)s',
                 level=logging.INFO
                )
    if log:
        assert not isinstance(id_,type(None))
        kwargs_['filename']=LOGGING_DIR+id_+'.log'
        logging.basicConfig(**kwargs_)
