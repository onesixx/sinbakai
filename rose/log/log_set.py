# log_set <-- log_cfg <-- log_color

import logging
import logging.config
import logging.handlers

import json
import atexit
import os
from pathlib import Path
from rose.config import BASE_DIR

# Use my own logger , not the root logger
logger = logging.getLogger("sixx_logger")

def setup_logging(log_filename: str = 'app.log'):
    curr_dir = os.path.dirname(__file__)
    config_file = os.path.join(curr_dir, 'log_cfg.json')
    with open(config_file) as f_in:
        config = json.load(f_in)

    LOG_DIR = BASE_DIR.joinpath('logs').resolve()
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, log_filename)
    config["handlers"]["file"]["filename"] = log_file

    logging.config.dictConfig(config)

    #Queue Handler for Non-blocking Logging
    # queue_handler = logging.getHandlerByName("queue_handler")
    # if queue_handler is not None:
    #     queue_handler.listener.start()
    #     atexit.register(queue_handler.listener.stop)
