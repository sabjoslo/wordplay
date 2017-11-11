from wordplay.utils import *

def test_to_ascii():
    nonascii_string=u"\xe9\x9d\x9eASCII\xe5\xad\x97\xe7\xac\xa6\xe4\xb8\xb2"
    assert to_ascii(nonascii_string)=='ASCII'
