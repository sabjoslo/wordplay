from collectors.collectors import InvalidUseException, Request
from collectors.utils import *

r=Request(log=False)

def test_get_data():
    r.get_data(url='https://graph.facebook.com/me?fields=id,name&access_token={}'.format(getAccessToken()))

    caught=False
    try:
        r.get_data(url='https://graph.facebook.com/me?fields=id,name')
    except Exception as e:
        assert isinstance(e, InvalidUseException)
        caught=True
        assert caught
    
    caught=False
    try:
        r.get_data()
    except Exception as e:
        assert isinstance(e, InvalidUseException)
        caught=True
        assert caught
