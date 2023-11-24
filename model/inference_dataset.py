from torch.utils.data import Dataset
from torch import tensor, float32, long
from pandas import read_csv
from cv2 import imread, imwrite
from numpy import array, uint8, zeros as numpy_zeros, ones as numpy_ones, all as numpy_all
from math import ceil

class CVInferenceDataset(Dataset):
    def __init__(self, meta_path, iter, trainer_data, matrix_shape=(256,256), transform=None):
        if trainer_data:
            self.meta = read_csv(meta_path+'metadata.csv')[iter:]
        else:
            self.meta = read_csv(meta_path+'metadata.csv')[:iter]
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

        n_h = ceil(image.shape[0] / self.matrix_shape[0]) * self.matrix_shape[0]
        n_w = ceil(image.shape[1] / self.matrix_shape[1]) * self.matrix_shape[1]

        target_res = numpy_zeros((target.shape[0], target.shape[1]), dtype=uint8)
        image_res = numpy_zeros((n_h, n_w, 3), dtype=uint8)
        image_res[:image.shape[0], :image.shape[1]] = image

        for i in range(self.class_count):
            color = self.color.iloc[i, 1:].tolist()
            mask = numpy_all(target == color, axis=-1)
            target_res[mask] = i

        target_result = numpy_ones((n_h, n_w), dtype=uint8)
        target_result[:target_res.shape[0], :target_res.shape[1]] = target_res

        def generate_parts(image_res, target_result, matrix_shape):
            for i in range(0, image_res.shape[0], matrix_shape[0] / 2):
                for j in range(0, image_res.shape[1], matrix_shape[1] / 2):
                    image_part = image_res[i:i+matrix_shape[0], j:j+matrix_shape[1]]
                    target_part = target_result[i:i+matrix_shape[0], j:j+matrix_shape[1]]
                    image_part = tensor(image_part, dtype=float32).permute(2, 0, 1) / 255
                    target_part = tensor(target_part, dtype=long)
                    yield image_part, target_part

        return index, generate_parts(image_res, target_result, self.matrix_shape)


    def imshow(self, image, target, output, image_name):
        image = array(image.permute(1,2,0) * 255).astype(uint8)
        # output = output.argmax(0)
        target_res = numpy_zeros(shape=(target.shape[0], target.shape[1], 3), dtype=uint8)
        # output_res = numpy_zeros(shape=(output.shape[0], output.shape[1], 3), dtype=uint8)
        for i in range(self.class_count):
            mask1 = target == i
            mask2 = output == i
            color = self.color.iloc[i, 1:].to_numpy(dtype=uint8)
            target_res[mask1] = color
            # output_res[mask2] = color

        height, width = image.shape[:2]
        composite_image = numpy_zeros((height, width * 3, 3), dtype=uint8)
        
        composite_image[:, :width] = image
        composite_image[:, width:width*2] = target_res
        # composite_image[:, width*2:] = output_res
        imwrite(f'image_res/{image_name}.jpg', composite_image)
