from rich.logging import RichHandler

import logging
def setup_logging(name):

    app_logger = logging.getLogger(name) 
    app_logger.setLevel(logging.INFO)
    
    handler = RichHandler(rich_tracebacks=True, markup=True)
    app_logger.addHandler(handler)
    return app_logger

