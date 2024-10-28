import cv2
import torch
import torchvision

from common.utils import ImageUtils


class BasicDatasetInfo(torch.utils.data.Dataset):
    def __init__(self, name, memo, path):
        self._name = name
        self._memo = memo

        self._dataset_type = None
        self._data_list: [dict] = self._load_data_list(path)

        self._to_tensor = torchvision.transforms.ToTensor()

    def _load_data_list(self, path) -> []:
        pass

    def get_name(self):
        return self._name

    def set_name(self, name: str):
        self._name = name

    def get_memo(self):
        return self._memo

    def set_memo(self, memo: str):
        self._memo = memo

    def get_type(self):
        return self._dataset_type

    def get_data_shape(self) -> tuple:
        pass

    def __len__(self):
        return len(self._data_list)


class BasicImageDatasetInfo(BasicDatasetInfo):
    def __init__(self, name, memo, path, width, height, ch):
        super().__init__(name, memo, path)
        self._width = width
        self._height = height
        self._ch = ch

    def get_width(self):
        return self._width

    def get_height(self):
        return self._height

    def get_ch(self):
        return self._ch

    def get_data_shape(self) -> tuple:
        data_shape = (self._ch, self._height, self._width)
        return data_shape

    def _load_image_resized(self, path):
        flag = cv2.IMREAD_GRAYSCALE if self._ch == 1 else cv2.IMREAD_COLOR

        image_ori = ImageUtils.imread(path, flag)
        image_resized = cv2.resize(image_ori, dsize=(self._width, self._height))

        if flag == cv2.IMREAD_COLOR:
            image_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

        return image_resized
