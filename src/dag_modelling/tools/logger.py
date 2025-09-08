from logging import (
    DEBUG,
    INFO,
    FileHandler,
    Formatter,
    Logger,
    StreamHandler,
    addLevelName,
    getLogger,
)
from sys import stdout

# To avoid a creation of duplicates save an instance
_loggers = {}


def get_logger(
    name="dag_modelling",
    *,
    filename: str | None = None,
    debug: bool = False,
    console: bool = True,
    # formatstr: Optional[str] = "%(asctime)s - %(levelname)s - %(message)s",
    formatstr: str | None = "%(levelname)s: %(message)s",
) -> Logger:
    if logger := _loggers.get(name):
        return logger
    logger = getLogger(name)

    level = DEBUG if debug else INFO
    logger.setLevel(level)
    formatter = Formatter(formatstr)
    if filename:
        fh = FileHandler(filename)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    if console:
        ch = StreamHandler(stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    _loggers[name] = logger
    return logger


def set_level(level, name="dag_modelling"):
    logger = _loggers[name]
    for handler in logger.handlers:
        handler.setLevel(level)
    logger.setLevel(level)


INFO1 = INFO - 1
INFO2 = INFO - 2
INFO3 = INFO - 3
addLevelName(INFO1, "INFO1")
addLevelName(INFO2, "INFO2")
addLevelName(INFO3, "INFO3")

info_levels = [INFO, INFO1, INFO2, INFO3, DEBUG]


def set_verbosity(verbosity: int, name="dag_modelling"):
    if verbosity<0:
        verbosity = 0

    try:
        level = info_levels[verbosity]
    except IndexError:
        level = info_levels[-1]

    logger = _loggers[name]
    for handler in logger.handlers:
        handler.setLevel(level)
    logger.setLevel(level)


logger = get_logger()
logger.propagate = False
