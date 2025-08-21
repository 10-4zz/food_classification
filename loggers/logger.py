import os
import sys
import logging

from .config import *


class LoggerController:
    _LOGGER_LEVELS = (logging.NOTSET, logging.DEBUG, logging.INFO, logging.WARNING, logging.WARN, logging.ERROR, logging.CRITICAL)
    """
    It is a singleton class that controls the logger instance.
        Args:
            log_file_name (str): The name of the log file. Default is "log.txt".
            output_dir (str): The directory to save the log file. Default is "./output".
    """
    def __init__(
            self,
            log_file_name: str = "log.txt",
            output_dir: str = "./output",
    ) -> None:

        self.log_file_name = log_file_name
        self.output_dir = output_dir

    def create_logger(
            self,
            logger_name: str = '__name__',
            log_level: int = logging.INFO,
            define_format: str = None,
            format_type: str = "default"
    ) -> logging.Logger:
        """
        Create a logger instance with the specified name.
            Args:
                logger_name (str): The name of the logger. Default is '__name__'.
                log_level (int): The logging level. Default is logging.INFO.
                define_format (str): A custom format string for the logger. Default is None.
                format_type (str): The format type for the logger. Default is 'default'.
            Returns:
                logging.Logger: The logger instance.
        """
        assert log_level in self._LOGGER_LEVELS , "Invalid log level provided."

        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)
        logger.propagate = False

        if define_format is not None:
            file_formatter, console_formatter = define_format
        else:
            file_formatter, console_formatter = get_logger_format(format_type=format_type)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(
            logging.Formatter(fmt=console_formatter, datefmt='%Y-%m-%d %H:%M:%S')
        )
        logger.addHandler(console_handler)

        file_handler = logging.FileHandler(os.path.join(self.output_dir, self.log_file_name), mode='a')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(fmt=file_formatter, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(file_handler)

        return logger

