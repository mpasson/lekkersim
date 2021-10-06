import logging
from solver import logger
import __main__
import solver as sv


logger.setLevel(logging.DEBUG)

std = logging.StreamHandler()
std.setLevel(logging.INFO)
fmt = logging.Formatter('%(levelname)s:%(name)s:%(message)s') 
std.setFormatter(fmt)


logger.addHandler(std)
logger.info(f'version {sv.__version__}')
if not sv.git_clean:
    logger.error('repository not clean')


class DuplicateFilter:
    """Filter for removing double entry in logger. Works on WARNING level or higher
    """
    def __init__(self):
        self.msgs = set()

    def filter(self, record):
        """Filtering function
        """
        if record.levelno < 30: return True
        rv = record.msg not in self.msgs
        self.msgs.add(record.msg)
        return rv

dup_filter = DuplicateFilter()
logger.addFilter(dup_filter)


def log_filter_on():
    """Activate filter to remove double log entries (WARNING and above)
    """
    logger.addFilter(dup_filter)

def log_filter_off():
    """Deactivate filter to remove double log entries
    """
    logger.removeFilter(dup_filter)



def logfile(filename = None, stdout = False, level = logging.INFO):
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
    logger.removeHandler(std)
    log_filter_off()
    logger.info(f'version {sv.__version__}')
    if not sv.git_clean:
        logger.error('repository not clean')
    log_filter_on()
    if stdout:
        logger.addHandler(std)
    

def debugfile(filename = None):
    """Set up and additional logfile for debug

    Args:
        filename (str): name of the file. Default is the name of the running python file + '.dbg'
    """
    filename = f'{__main__.__file__}.dbg' if filename is None else filename
    han = logging.FileHandler(filename, mode='w')
    han.setLevel(logging.DEBUG)
    han.setFormatter(fmt)
    logger.addHandler(han)
    
            
