import logging
from solver import logger
import __main__


logger.setLevel(logging.DEBUG)

std = logging.StreamHandler()
std.setLevel(logging.WARNING)
fmt = logging.Formatter('%(levelname)s:%(name)s:%(message)s') 
std.setFormatter(fmt)


logger.addHandler(std)


def logfile(filename = None, stdout = False, level = logging.WARNING):
    """Switch log output to file

    Args:
        filename (str): name of the file. Default is the name of the running python file + '.log'
        std (bool): if True, output to stdout is also kept. Default is false
        level (int): level of the file handler, default is logging.WARNING (30)
    """
    filename = f'{__main__.__file__}.log' if filename is None else filename
    han = logging.FileHandler(filename, mode='w')
    han.setFormatter(fmt)
    logger.addHandler(han)
    han.setLevel(level)
    if not stdout:
        logger.removeHandler(std)
    

def debugfile(filename = None):
    """Set up and additional logfile for debug

    Args:
        filename (str): name of the file. Default is the name of the running python file + '.dbg'
    """
    filename = f'{__main__.__file__}.dbg' if filename is None else filename
    han = logging.FileHandler(filename, mode='w')
    han.setLevel(logging.DEBUG)
    logger.addHandler(han)
    
            
