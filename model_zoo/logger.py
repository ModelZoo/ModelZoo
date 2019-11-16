from loguru import logger
from os.path import dirname, exists, join
from os import makedirs


def get_logger(config):
    """
    Get logger from loguru.
    :param flags:
    :return:
    """
    if not config.get('log_enable'):
        return logger
    
    log_path = None
    
    if config.get('log_path'):
        log_path = config.get('log_path')
    
    if not log_path:
        log_path = join(config.get('log_folder'), config.get('log_file'))
    
    # check folder
    if not exists(dirname(log_path)):
        makedirs(dirname(log_path))
    
    # init kwargs
    kwargs = {}
    kwargs['level'] = config.get('log_level')
    kwargs['sink'] = log_path
    
    if config.get('log_format'):
        kwargs['format'] = config.get('log_format')
    
    if config.get('log_rotation'):
        kwargs['rotation'] = config.get('log_rotation')
    
    if config.get('log_retention'):
        kwargs['retention'] = config.get('log_retention')
    
    # add processor
    logger.add(**kwargs)
    
    return logger
