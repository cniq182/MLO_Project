from loguru import logger
import sys

logger.remove()

# Terminal: warnings+
logger.add(sys.stderr, level="WARNING")

# File: everything
logger.add("my_log.log", level="DEBUG", rotation="100 MB")
