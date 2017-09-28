import logging
import json
import os
import random
import requests
import string
import time
from config import *
from utils import *

# A wrapper for functions to interact directly with the Graph API
class Request():
    def __init__(self, log=True, id_=None, base_url=None):
        if isinstance(id_,type(None)):
            id_=''.join(random.choice(string.lowercase) for _ in range(6))
        self.id_=id_
        startLog(log=log, id_=id_)
        self.BASE_URL=base_url
    
    # Returns a json object with the response to a query whose arguments
    # are specified as arguments to the function.
    def get_data(self,url=None,**kwargs):
        if not url:
            url=self._format_request(**kwargs)
        while True:
            try:
                logging.info('Sending request to '+url)
                response=requests.get(url)
                assert response.status_code==200
                return json.loads(response.content)
            except Exception as e:
                if isinstance(e,AssertionError):
                    retval=self._failed_request_handler(response.content)
                    if not isinstance(retval, type(None)):
                        return retval
                    logging.info("""
    Request failed with response {}: {}. Retrying.
    """.format(response.status_code,response.content))
                    continue
                logging.info("""
    requests failed to make request with error {}. Retrying.
    """.format(e))
                time.sleep(1)
                continue

    def _failed_request_handler(self, content_):
        raise InvalidUseException

    def _format_request(self):
        raise InvalidUseException

class InvalidUseException(Exception):
    def __init__(self):
        super(Exception, self).__init__("""
This is a stand-in function that should never be called. Call one of the
inheriting collectors's function of the same name, or overwrite this function
with a custom one.
""")

class FacebookCollector(Request):
    def __init__(self, log=True, id_=None):
        Request.__init__(self, log=log, id_=id_)
        self.BASE_URL='https://graph.facebook.com/'

    def _failed_request_handler(self, content_):
        if '#17' in content_:
            logging.info('Rate limit reached. Wait one hour.')
            time.sleep(3600)
        if '#100' in content_:
            logging.info("""
            Reached absolute limit. Stop here.
            
            For more information, visit
            https://developers.facebook.com/bugs/1772936312959430/.
            """)
            return [] 

    # A helper function that formats API queries whose arguments are
    # specified as arguments to the function.
    def _format_request(self,node,edge=None,**kwargs):
        edge='' if isinstance(edge,type(None)) else edge+'/'
        header_str='{}/{}?'.format(node,edge)
        access_token_str='&access_token='+getAccessToken()
        header_str+='&'.join('{}={}'.format(k,v) for k,v in kwargs.items())
        header_str+=access_token_str
        return self.BASE_URL+header_str

    # Retrieve the index of the next "page" of results (used to paginate
    # through multiple pages of results)
    def get_next_val(self,json_obj,page=False):
        if 'next' not in json_obj['paging']:
            return False
        nextval=json_obj['paging']['next']
        if page:
            return self.get_data(url=nextval)
        return nextval
