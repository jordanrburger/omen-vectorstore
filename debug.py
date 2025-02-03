import sys
import traceback
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def exception_hook(exctype, value, tb):
    """Custom exception hook to log uncaught exceptions."""
    logger.error("Uncaught exception:", exc_info=(exctype, value, tb))
    traceback.print_exception(exctype, value, tb)

sys.excepthook = exception_hook

from app.main import cli

if __name__ == "__main__":
    try:
        cli.main(args=["index", "--tenant-id", "1255"])
    except Exception as e:
        logger.error("Error running CLI:", exc_info=True) 