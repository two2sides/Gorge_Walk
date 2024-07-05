import logging
import datetime
import json


def get_F(path="arena/conf/gorge_walk/F_level_0.json"):
    with open(path, "r") as f:
        F = json.load(f)
    return F


def get_logger(level=logging.INFO, name='default'):
    # Configure logging to save logs to a file
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    logging.basicConfig(filename=f'log/{name}-{timestamp}.log')
    
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create a console handler and set the formatter
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Add the console handler to the logger
    logger.addHandler(console_handler)
    
    return logger
