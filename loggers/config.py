from typing import Tuple, Optional

from dataclasses import dataclass
from termcolor import colored


@dataclass(frozen=True)
class LoggerFormatConfig:
    DEFAULT_FORMAT: str = '[%(asctime)s %(name)s] (file: %(filename)s func: %(funcName)s line: %(lineno)d): %(levelname)s %(message)s'
    DEFAULT_COLOR_FORMAT: str = colored('[%(asctime)s %(name)s]', 'green') + \
                                 colored(' (file: %(filename)s func: %(funcName)s line: %(lineno)d)', 'yellow') + \
                                    ': %(levelname)s %(message)s'

    JUST_MESSAGE_FORMAT: str = '%(message)s'
    INFO_MESSAGE_FORMAT: str = '%(levelname)s %(message)s'

    TIME_INFO_MESSAGE_FORMAT: str = '[%(asctime)s] %(levelname)s %(message)s'
    TIME_INFO_MESSAGE_COLOR_FORMAT: str = colored('[%(asctime)s]', 'green') + \
                                           '%(levelname)s %(message)s'

    LOGGER_NAME_FORMAT: str = '[%(asctime)s %(name)s] %(levelname)s %(message)s'
    LOGGER_NAME_COLOR_FORMAT: str = colored('[%(asctime)s %(name)s]', 'green') + \
                                     ': %(levelname)s %(message)s'

    FULL_LOGGER_FORMAT: str = '[%(asctime)s %(name)s] (file: %(filename)s func: %(funcName)s line: %(lineno)d): %(levelname)s %(message)s'
    FULL_LOGGER_COLOR_FORMAT: str = colored('[%(asctime)s %(name)s]', 'green') + \
                                    colored(' (file: %(filename)s func: %(funcName)s line: %(lineno)d)', 'yellow') + \
                                    ': %(levelname)s %(message)s'


def get_logger_format(format_type: str = 'default') -> Optional[Tuple[str, str]]:
    """
    Args:
        format_type (LoggerFormatType): The type of logger format to retrieve.
    """
    if format_type == 'default':
        return LoggerFormatConfig.DEFAULT_FORMAT, LoggerFormatConfig.DEFAULT_COLOR_FORMAT
    elif format_type == 'just_message':
        return LoggerFormatConfig.JUST_MESSAGE_FORMAT, LoggerFormatConfig.JUST_MESSAGE_FORMAT
    elif format_type == 'info_message':
        return LoggerFormatConfig.INFO_MESSAGE_FORMAT, LoggerFormatConfig.INFO_MESSAGE_FORMAT
    elif format_type == 'time_info_message':
        return LoggerFormatConfig.TIME_INFO_MESSAGE_FORMAT, LoggerFormatConfig.TIME_INFO_MESSAGE_COLOR_FORMAT
    elif format_type == 'logger_name':
        return LoggerFormatConfig.LOGGER_NAME_FORMAT, LoggerFormatConfig.LOGGER_NAME_COLOR_FORMAT
    elif format_type == 'full':
        return LoggerFormatConfig.FULL_LOGGER_FORMAT, LoggerFormatConfig.FULL_LOGGER_COLOR_FORMAT
    else:
        raise TypeError(
            f"Invalid format type: {format_type}. "
            "Available options are: 'default', 'just_message', 'info_message', "
            "'time_info_message', 'logger_name', 'full'."
        )
