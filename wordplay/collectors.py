from copy import deepcopy
import logging
import json
import os
import random
import requests
import string
import time
from wordplay.config import *
from wordplay.utils import *

# A wrapper for functions to interact directly with the Graph API
class Request():
    def __init__(self, log=True, id_=None, base_url=None, **kwargs):
        if isinstance(id_,type(None)):
            id_=''.join(random.choice(string.lowercase) for _ in range(6))
        self.id_=id_
        startLog(log=log, id_=id_)
        self.BASE_URL=base_url
        if 'wait' in kwargs.keys():
           self.wait=kwargs['wait']
        if 'forum' in kwargs.keys():
            self.forum=kwargs['forum']
        if 'include' in kwargs.keys():
            self.include=kwargs['include']
    
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
                return response
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
            return 1

    # A helper function that formats API queries whose arguments are
    # specified as arguments to the function.
    def _format_request(self,node,edge=None,**kwargs):
        edge='' if isinstance(edge,type(None)) else edge+'/'
        header_str='{}/{}?'.format(node,edge)
        access_token_str='&access_token='+getPublicKey()
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

class StackExchangeCollector(Request):
    def __init__(self, log=True, id_=None):
        Request.__init__(self, log=log, id_=id_)
        self.BASE_URL='https://api.stackexchange.com/2.2/'

    def _failed_request_handler(self, content_):
        raise Exception(content_) 

    def _format_request(self,objects,**kwargs):
        header_str='/'.join(objects)+'?'
        header_str+='&'.join('{}={}'.format(k,v) for k,v in kwargs.items())
        return self.BASE_URL+header_str
 
class DisqusCollector(Request):
    def __init__(self, forum, include=['unapproved','approved','flagged',
                 'highlighted'], wait=False, log=True, id_=None):
        Request.__init__(self, forum=forum, include=include, wait=wait, 
                         log=log, id_=id_)
        self.BASE_URL="https://disqus.com/api/3.0/"
        self.rate_limit=dict(limit=None,remaining=None,reset=None)
    
    def get_data(self,resource,output_type,**kwargs):
        if self.rate_limit['remaining']==1:
            if not self.wait:
                raise Exception('Rate limit exceeded.')
            else:
                sleeptime=self.rate_limit['reset']-time.time()+60 # Buffer,
                # just in case
                logging.info('Sleeping for {} seconds.'.format(sleeptime))
                time.sleep(sleeptime)
        kwargs_=deepcopy(kwargs)
        kwargs_['resource']=resource
        kwargs_['output_type']=output_type
        response=Request.get_data(self, **kwargs_)

        # Update rate limit status
        resp_headers=response.headers
        self.rate_limit['limit']=int(resp_headers['X-Ratelimit-Limit'])
        self.rate_limit['remaining']=int(resp_headers['X-Ratelimit-Remaining'])
        self.rate_limit['reset']=float(resp_headers['X-Ratelimit-Reset'])

        return response

    def _format_request(self,resource,output_type,**kwargs):
        if resource=='trends' and 'limit' in kwargs:
            if kwargs['limit']>10:
                logging.info('Resetting limit to 10 (maximum when asking for trending threads).')
                kwargs['limit']=10
        header_str='{}/{}.json?'.format(resource,output_type)
        public_key_str='&api_key='+getPublicKey()
        if resource=='threads':
            assert 'thread' in kwargs
        if resource=='users':
            assert 'user' in kwargs
        request_args=[]
        for k,v in kwargs.items():
            if not hasattr(v,'__iter__'):
                request_args.append((k,v))
            else:
                for v_ in v:
                    request_args.append((k,v_))
        header_str+='&'.join('{}={}'.format(k,v) for k,v in request_args)
        header_str+=public_key_str
        return self.BASE_URL+header_str

    # Retrieve the index of the next "page" of results (used to paginate
    # through multiple pages of results)
    def get_next_val(self,data):
        if data['cursor']['hasNext']:
            return data['cursor']['next']
        else:
            return None
