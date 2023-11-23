from torch.utils.data import Dataset
from torch import tensor, zeros, float64, float32, long
from pandas import read_csv
from cv2 import resize, imread, imwrite
from numpy import array, uint8, zeros as numpy_zeros

class CVDataset(Dataset):
    def __init__(self, meta_path, matrix_shape=None, transform=None):
        self.meta = read_csv(meta_path+'metadata.csv')
        self.meta = self.meta[self.meta['split'] == 'train'].drop('split', axis=1)
        self.meta_path = meta_path
        self.color = read_csv(meta_path+'class_dict.csv')
        self.class_count = len(self.color)
        self.matrix_shape = matrix_shape
        self.transform = transform

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        index, image_path, target_path = self.meta.iloc[idx]
        image_path = self.meta_path + image_path
        target_path = self.meta_path + target_path

        image = imread(image_path)
        target = imread(target_path)
        if self.matrix_shape:
            image = resize(image, self.matrix_shape)
            target = resize(target, self.matrix_shape)

        target_res = numpy_zeros(shape=(target.shape[0], target.shape[1]), dtype=uint8)

        for i in range(self.class_count):
            mask = (target == self.color.iloc[i, 1:].tolist()).all(axis=-1)
            target_res[mask] = i

        image = tensor(image, dtype=float32).permute(2, 0, 1) / 255
        target_res = tensor(target_res, dtype=long)
        return index, image, target_res

    def imshow(self, image, target, output, image_name):
        image = array(image.permute(1,2,0) * 255).astype(uint8)
        output = output.argmax(0)
        target_res = numpy_zeros(shape=(target.shape[0], target.shape[1], 3), dtype=uint8)
        output_res = numpy_zeros(shape=(output.shape[0], output.shape[1], 3), dtype=uint8)
        for i in range(self.class_count):
            mask1 = target == i
            mask2 = output == i
            color = self.color.iloc[i, 1:].to_numpy(dtype=uint8)
            target_res[mask1] = color
            output_res[mask2] = color

        height, width = image.shape[:2]
        composite_image = numpy_zeros((height, width * 3, 3), dtype=uint8)
        
        composite_image[:, :width] = image
        composite_image[:, width:width*2] = target_res
        composite_image[:, width*2:] = output_res

        imwrite(f'image_res/{image_name}.jpg', composite_image)