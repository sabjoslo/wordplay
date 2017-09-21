from collectors.collectors import FacebookCollector
from collectors.utils import *

api_bind=FacebookCollector(log=False)

def test_format_request():
    url=api_bind._format_request(node='me',fields='id,name')
    assert url=='https://graph.facebook.com/me/?fields=id,name&access_token={}'.format(getAccessToken())

    url=api_bind._format_request(node='cnn',edge='posts')
    assert url=='https://graph.facebook.com/cnn/posts/?&access_token={}'.format(getAccessToken())

def test_get_data():
    url=api_bind._format_request(node='cnn',edge='posts')
    r0=api_bind.get_data(url)
    r1=api_bind.get_data(node='cnn',edge='posts')
    assert r0==r1

def test_exceptions():
    assert api_bind._failed_request_handler('#100')==[]
    
