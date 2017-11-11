import json
from wordplay.collectors import FacebookCollector
from wordplay.utils import *

api_bind=FacebookCollector(log=False)

def test_format_request():
    url=api_bind._format_request(node='me',fields='id,name')
    assert url=='https://graph.facebook.com/me/?fields=id,name&access_token={}'.format(getPublicKey())

    url=api_bind._format_request(node='cnn',edge='posts')
    assert url=='https://graph.facebook.com/cnn/posts/?&access_token={}'.format(getPublicKey())

def test_get_data():
    url=api_bind._format_request(node='cnn',edge='posts')
    r0=api_bind.get_data(url)
    r1=api_bind.get_data(node='cnn',edge='posts')
    j0=json.loads(r0.content)
    j1=json.loads(r1.content)
    assert j0==j1

def test_exceptions():
    assert api_bind._failed_request_handler('#100')==[]
    
