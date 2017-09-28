import os
from setuptools import setup

config_str="""[settings]

# Where your app's ID is stored in plain text
APP_ID_FILE=''

# Where your app's secret key is stored in plain text
APP_SECRET_FILE=''

# Where your API access token is stored in plain text
ACCESS_TOKEN_FILE=''

# User agent string to include in all requests to the API
USER_AGENT=''

# Log files will be stored in LOGGING_DIR.
LOGGING_DIR='./'
"""

if not os.path.exists(os.path.expanduser('~/.collectors_config')):
    with open(os.path.expanduser('~/.collectors_config'), 'w') as fh:
        fh.write(config_str)

setup(name='collectors',
      version='1.0.0.dev1',
      description="""A toolbox for retrieving natural language data from
a variety of publicly-accessible platforms.""",
      url='https://github.com/sabjoslo/collectors',
      author='Sabina Sloman',
      author_email='sabina.joslo@gmail.com',
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Researchers',
        'Programming Language :: Python :: 2'
      ],
      python_requires='<3.0'
     )
