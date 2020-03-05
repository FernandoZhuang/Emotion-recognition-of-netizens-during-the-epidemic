import os
import configparser

#region Directory
cfg = configparser.ConfigParser()
conf_path = os.path.dirname(os.path.realpath(__file__)+'/config.cfg')
cfg.read(conf_path)
#endregion