import pickle
import numpy as np
from PIL import Image

class ImageRecord:
    def __init__(self, label, image: np.array):
        self._label = label
        self._data = image.tobytes()
        self._channels = image.shape[2] if len(image.shape) > 2 else 1
        self._size = image.shape[:2]

    @property
    def label(self):
        return self._label

    @property
    def image(self):
        img = np.frombuffer(self._data, dtype=np.uint8)
        return img.reshape(*self._size, self._channels)

    def dumps(self):
        return pickle.dumps(self)

    def __repr__(self):
        return f'{self.__class__.__name__} (channels={self._channels} size={self._size} len={len(self._data)})'

    @staticmethod
    def loads(data):
        return pickle.loads(data)

    @staticmethod
    def from_image(label, file: str):
        data = np.array(Image.open(file), np.uint8)
        return ImageRecord(label, data)
