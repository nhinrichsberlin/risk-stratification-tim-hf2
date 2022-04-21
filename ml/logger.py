import logging
import datetime

time_start = str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")).strip()


def custom_logger(file: str, output_path: str = f'logs/{time_start.replace(" ", "_")}.log'):
    logger = logging.getLogger(file)
    logger.setLevel(logging.INFO)

    if output_path is not None:
        logging.basicConfig(filename=output_path)

    # avoid duplications when running the same thing multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # some formatting
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', "%Y-%m-%d %H:%M:%S")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
