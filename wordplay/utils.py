import logging
import os
import random
import re
import string
import time
from wordplay.config import settings
PUBLIC_KEY_FILE=settings['public_key_file']
APP_ID_FILE=settings['app_id_file']
SECRET_KEY_FILE=settings['secret_key_file']
LOGGING_DIR=settings['logging_dir']

def getPublicKey():
    if not os.path.exists(PUBLIC_KEY_FILE):
        return raw_input('Enter public key here: ').strip()
    with open(PUBLIC_KEY_FILE,'r') as fh:
        return fh.read().strip()

def getAppID():
    if not os.path.exists(APP_ID_FILE):
        return raw_input('Enter app ID here: ').strip()
    with open(APP_ID_FILE,'r') as fh:
        return fh.read().strip()

def getSecretKey():
    if not os.path.exists(SECRET_KEY_FILE):
        return raw_input('Enter secret key here: ').strip()
    with open(SECRET_KEY_FILE,'r') as fh:
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
    if log:
        kwargs_=dict(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO
                    )
        assert not isinstance(id_,type(None))
        filename=LOGGING_DIR+id_+'.log'
        fh=os.open(filename, os.O_CREAT, 0o600)
        os.close(fh)
        kwargs_['filename']=LOGGING_DIR+id_+'.log'
        logging.basicConfig(**kwargs_)
    else:
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

def iso8601_to_unix(iso8601_str):
    return time.mktime(time.strptime(iso8601_str,'%Y-%m-%dT%H:%M:%S'))

def unix_to_iso8601(unix_str):
    return time.strftime('%Y-%m-%dT%H:%M:%S',time.localtime(unix_str))

# Encode a non-ASCII string to ASCII. If a non-ASCII character is encountered,
# delete it.
def to_ascii(s):
    return s.encode('ascii', 'ignore')
