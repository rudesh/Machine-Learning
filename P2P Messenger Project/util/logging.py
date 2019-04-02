import logging

LEVEL = logging.DEBUG
FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'

def setup():
    # logging.basicConfig(level=LEVEL, format=FORMAT) # doesn't work on server for some reason

    logger = logging.getLogger()
    logger.setLevel(LEVEL)

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(FORMAT))
    logger.handlers.clear() # avoid duplicating handlers
    logger.addHandler(handler)
