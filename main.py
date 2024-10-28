import logging

from common.utils import GeneralUtils

GeneralUtils.set_seed(99)


def logger_setup():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s >> %(message)s", "%Y-%m-%d %H:%M:%S")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler("./log.txt")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def main():
    logger_setup()

    from image_matching.ui.hardcoded import HardCodedUI as IMUI
    IMUI().train_alto()
    IMUI().test_alto()


if __name__ == '__main__':
    main()
