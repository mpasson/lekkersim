import logging
from solver import logger
import __main__


logger.setLevel(logging.DEBUG)

std = logging.StreamHandler()
std.setLevel(logging.WARNING)
fmt = logging.Formatter('%(levelname)s:%(name)s:%(message)s') 
std.setFormatter(fmt)


logger.addHandler(std)


def logfile(filename = f'{__main__.__file__}.log', stdout = False, level = logging.WARNING):
    """Switch log output to file

    Args:
        filename (str): name of the file. Default is the name of the running python file + '.log'
        std (bool): if True, output to stdout is also kept. Default is false
        level (int): level of the file handler, default is logging.WARNING (30)
    """
    han = logging.FileHandler(filename, mode='w')
    han.setFormatter(fmt)
    logger.addHandler(han)
    if not stdout:
        logger.removeHandler(std)
    

def debug_file(filename = f'{__main__.__file__}.dbg'):
    """Set up and additional logfile for debug

    Args:
        filename (str): name of the file. Default is the name of the running python file + '.dbg'
    """
    han = logging.FileHandler(filename, mode='w')
    han.setLevel(logging.DEBUG)
    logger.addHandler(han)
    
            
