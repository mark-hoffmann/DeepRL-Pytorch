import logging
import os
import sys

class Logger():

    def __init__(self, level):
        logging.basicConfig(stream=sys.stdout)
        self.logger = logging.getLogger("logger")
        self.setLevel(level)

    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def setLevel(self, level):
        if level not in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            raise Exception("Invalid logging level")

        if level == "DEBUG":
            self.logger.setLevel(logging.DEBUG)
        elif level == "INFO":
            self.logger.setLevel(logging.INFO)
        elif level == "WARNING":
            self.logger.setLevel(logging.WARNING)
        elif level == "ERROR":
            self.logger.setLevel(logging.ERROR)


level = os.environ.get("LOG_LEVEL", "DEBUG")

log = Logger(level)
