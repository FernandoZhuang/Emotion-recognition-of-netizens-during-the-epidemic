import os
import time
import datetime
import configparser

# region Directory
cfg = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
conf_path = os.path.dirname(os.path.realpath(__file__)) + '/config.cfg'
cfg.read(conf_path)


# endregion

# region 时间
def timestamp_2_readable(time_stamp=time.time()):
    """
    时间戳转换为可读时间
    :param time_stamp: 时间戳，当前时间：time.time()
    :return: 可读时间字符串
    """
    return datetime.datetime.fromtimestamp(time_stamp).strftime('%Y-%m-%d %H:%M:%S')


# endregion

# region 文件生成
def make_document(flag: str, content: list, is_delete=False):
    '''
    防止覆盖文件
    :return:
    '''
    file_name = cfg.get('COMMON', 'output_path') + '/' + flag + '_' + timestamp_2_readable()
    with open(file_name, 'w') as f:
        for item in content:
            f.writelines(str(item))

# endregion
