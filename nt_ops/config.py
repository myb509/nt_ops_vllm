import os
import logging
import inspect

logger = logging.getLogger("NTOPS")


def _parse_bool_env_var(env_var: str, default: str = "false") -> bool:
    """Parses a boolean environment variable."""
    return os.getenv(env_var, default).lower() in ["true", "1"]


def _parse_int_env_var(env_var: str, default: int = 1) -> int:
    """Parse a int environment variable."""
    return int(os.getenv(env_var, default))


NT_DEBUG = _parse_bool_env_var("NT_DEBUG", "false")
NT_OP = _parse_bool_env_var("NT_OP", "true")
NT_MAX_NUM_CONFIG = _parse_int_env_var("NT_MAX_NUM_CONFIG", 2)
NT_STATIC_MODE = _parse_bool_env_var("NT_STATIC_MODE", "true")


class LogColors:
    RED = "\033[31m"
    RESET = "\033[0m"


logger.info(f"{LogColors.RED}NT DEBUG: {NT_DEBUG}{LogColors.RESET}")


def debug_log(info: str):
    """
    Logs a debug message if NT_DEBUG is enabled.
    The log message will be prefixed with the caller's filename.
    """
    if NT_DEBUG:
        # stack()[1] gives the frame of the caller.
        caller_frame = inspect.stack()[1]
        caller_filename = os.path.basename(caller_frame.filename)
        logger.info(f"{LogColors.RED}[{caller_filename}] {info}{LogColors.RESET}")