"""
Define logger for this project.
"""
import os
import sys
import time
import traceback
from typing import Optional, Union, Dict, Any


text_colors = {
    "logs": "\033[34m",   # 蓝色
    "info": "\033[32m",   # 绿色
    "warning": "\033[33m",# 黄色
    "debug": "\033[93m",  # 亮黄
    "error": "\033[31m",  # 红色
    "bold": "\033[1m",    # 加粗
    "end_color": "\033[0m",  # 重置颜色
    "light_red": "\033[36m",
}

# LOG LEVEL
DEBUG = 0
WARNING = 1
INFO = 2
LOG = 3

SAVE_LOG = False
SHOW_TIME = False

LOG_PATH = os.environ.get('log_path', './output/log')
LOG_FILE_NAME = None

LOG_LEVEL = 2


def set_logger(log_config: Dict[str, Any]) -> None:
    global SAVE_LOG
    global SHOW_TIME
    global LOG_PATH
    global LOG_LEVEL
    global LOG_FILE_NAME

    if 'save_log' in log_config:
        SAVE_LOG = log_config['save_log']
    if 'show_time' in log_config:
        SHOW_TIME = log_config['show_time']
    if 'log_path' in log_config:
        LOG_PATH = log_config['log_path']
    if 'log_file_name' in log_config:
        LOG_FILE_NAME = log_config['log_file_name']
    else:
        LOG_FILE_NAME = get_curr_time_stamp()
    if 'log_level' in log_config:
        LOG_LEVEL = log_config['log_level']
    
    if SAVE_LOG:
        os.makedirs(LOG_PATH, exist_ok=True)
        info(message=f"The log file will be saved in the file '{os.path.join(LOG_PATH, f'{LOG_FILE_NAME}.log')}'.")
    else:
        info(message="This run will not save logs locally.")

def get_curr_time_stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def error(message: str) -> None:
    time_stamp = get_curr_time_stamp()
    error_str = (
        text_colors["error"]
        + text_colors["bold"]
        + "ERROR  "
        + text_colors["end_color"]
    )

    if sys.exc_info()[0] is None:
        traceback.print_stack()
    else:
        traceback.print_exc()
    sys.exit("{} - {} - {}. Exiting!!!".format(time_stamp, error_str, message))


def color_text(in_text: str) -> str:
    return text_colors["light_red"] + in_text + text_colors["end_color"]


def log(message: str) -> None:
    time_stamp = get_curr_time_stamp()
    log_str = (
        text_colors["logs"] + text_colors["bold"] + "LOGS   " + text_colors["end_color"]
    )
    if LOG_LEVEL <= LOG:
        if SHOW_TIME:
            print("{} - {} - {}".format(time_stamp, log_str, message))
        else:
            print("{} - {}".format(log_str, message))
        if SAVE_LOG:
            with open(os.path.join(LOG_PATH, f"{LOG_FILE_NAME}.log"), 'a') as f:
                f.writelines('LOGS - ' + message + '\n')


def debug(message: str) -> None:
    time_stamp = get_curr_time_stamp()
    debug_str = (
        text_colors["debug"]
        + text_colors["bold"]
        + "DEBUG   "
        + text_colors["end_color"]
    )
    if LOG_LEVEL <= DEBUG:
        if SHOW_TIME:
            print("{} - {} - {}".format(time_stamp, debug_str, message))
        else:
            print("{} - {}".format(debug_str, message))
        if SAVE_LOG:
            with open(os.path.join(LOG_PATH, f"{LOG_FILE_NAME}.log"), 'a') as f:
                f.writelines('DEBUG - ' + message + '\n')


def info(message: str, print_line: Optional[bool] = False) -> None:
    time_stamp = get_curr_time_stamp()
    info_str = (
        text_colors["info"] + text_colors["bold"] + "INFO   " + text_colors["end_color"]
    )
    if LOG_LEVEL <= INFO:
        if SHOW_TIME:
            print("{} - {} - {}".format(time_stamp, info_str, message))
        else:
            print("{} - {}".format(info_str, message))
        if print_line:
            double_dash_line(dashes=150)
        if SAVE_LOG:
            with open(os.path.join(LOG_PATH, f"{LOG_FILE_NAME}.log"), 'a') as f:
                f.writelines('INFO - ' + message + '\n')


def warning(message: Union[str, Warning]) -> None:
    time_stamp = get_curr_time_stamp()
    if isinstance(message, Warning):
        message = f"{type(message).__name__}({','.join(map(repr, message.args))}"

    warn_str = (
        text_colors["warning"]
        + text_colors["bold"]
        + "WARNING"
        + text_colors["end_color"]
    )
    if LOG_LEVEL <= WARNING:
        if SHOW_TIME:
            print("{} - {} - {}".format(time_stamp, warn_str, message))
        else:
            print("{} - {}".format(warn_str, message))
        if SAVE_LOG:
            with open(os.path.join(LOG_PATH, f"{LOG_FILE_NAME}.log"), 'a') as f:
                f.writelines('WARNING - ' + message + '\n')


def double_dash_line(dashes: Optional[int] = 75) -> None:
    print(text_colors["error"] + "=" * dashes + text_colors["end_color"])


def singe_dash_line(dashes: Optional[int] = 67) -> None:
    print("-" * dashes)
