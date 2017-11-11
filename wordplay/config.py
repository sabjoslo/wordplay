import ConfigParser
import os

CONFIG_FILE=os.path.expanduser('~/.wordplay_config')

configParser=ConfigParser.RawConfigParser()
configParser.read(CONFIG_FILE)
settings=dict(configParser.items('settings'))

