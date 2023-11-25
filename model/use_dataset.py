from torch.utils.data import Dataset
from torch import tensor, float32, long
from pandas import read_csv
from cv2 import imread, imwrite
from numpy import array, uint8, zeros as numpy_zeros, ones as numpy_ones, all as numpy_all, float32 as numpy_float32, int64
from math import ceil

class UseDataset(Dataset):
    def __init__(self, meta_path, matrix_shape=(256,256), transform=None):
        self.meta = read_csv(meta_path+'metadata2.csv')
        self.meta = self.meta[self.meta['split'] == 'train'].drop('split', axis=1)
        self.meta_path = meta_path
        self.color = read_csv(meta_path+'class_dict.csv')
        self.class_count = len(self.color)
        self.matrix_shape = matrix_shape
        self.transform = transform

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        index, image_path = self.meta.iloc[idx]
        image_path = self.meta_path + image_path

        image = imread(image_path)
        image_shape = (image.shape[0], image.shape[1])

        n_h = ceil(image.shape[0] / self.matrix_shape[0]) * self.matrix_shape[0]
        n_w = ceil(image.shape[1] / self.matrix_shape[1]) * self.matrix_shape[1]

        image_res = numpy_zeros((n_h, n_w, 3), dtype=uint8)
        image_res[:image.shape[0], :image.shape[1]] = image

        data = []
        for i in range(0, image_res.shape[0], self.matrix_shape[0]):
            for j in range(0, image_res.shape[1], self.matrix_shape[1]):
                image_part = image_res[i:i+self.matrix_shape[0], j:j+self.matrix_shape[1]].transpose(2, 0, 1).astype(numpy_float32) / 255
                data.append((image_part))
        return image_shape, data