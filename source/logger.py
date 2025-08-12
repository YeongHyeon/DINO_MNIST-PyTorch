import logging
from logging import Logger

class LoggerSingleton:
    _instance = None

    @staticmethod
    def get_logger() -> Logger:
        if LoggerSingleton._instance is None:

            logger = logging.getLogger('logger')
            logger.setLevel(logging.DEBUG)

            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)

            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)

            if not logger.hasHandlers():
                logger.addHandler(console_handler)

            LoggerSingleton._instance = logger

        return LoggerSingleton._instance