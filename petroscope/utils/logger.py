from loguru import logger
import sys

logger.remove()  # remove default logger

# add colored console logging
logger.add(
    sys.stderr,
    format="<green>{time}</green> <level>{level}</level> <cyan>{message}</cyan>",
    level="INFO",
    colorize=True,  # ensures color output
)

# add file logging (without color)
logger.add("petroscope.log", rotation="10 MB", level="DEBUG")
