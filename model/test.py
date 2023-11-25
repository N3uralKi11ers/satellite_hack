from dataset import use_dataset
from deeplabv3 import DeepLabV3
# from unet import UNet
# from unetplusplus import UNetPlusPlus
from torch import no_grad, sigmoid
import numpy as np
from torch.utils.data import DataLoader
from cv2 import imread, imwrite

data = UseDataset('/home/denis/code/PatternCV/', (1024,1024))

# model = DeepLabV3(2)
# model = UNet(3, 2)
model = DeepLabV3(2).to(device)

model.eval()
with no_grad():
    for k in range(len(data))
        image_shape, dat = data[i]
        count_n = len(dat)
        loader = DataLoader(dat, batch_size=batch_size, shuffle=False, num_workers=cpu_count(), pin_memory=False)
        part_output = []
        for x, y in loader:
            images, targets = x.to(device), y.to(device)
            optim.zero_grad()
            output = (model(images).softmax(1)[:, 0] > 0.6).detach().cpu().numpy()
            for j in range(len(output)):
                part_output.append(output[j])
        part_output = np.array(part_output)

        reconstructed_image = np.zeros(image_shape, dtype=np.uint8)
        for i in range(0, reconstructed_image.shape[0], matrix_shape[0]):
            for j in range(0, reconstructed_image.shape[1], matrix_shape[1]):
                image_part_tensor, _ = part_output[0]
                part_output[1:] = part_output
                image_part = (image_part_tensor * 255).astype(np.uint8).transpose(1, 2, 0)
                reconstructed_image[i:i+matrix_shape[0], j:j+matrix_shape[1]] = image_part

        imwrite(f'./image_res/test_mask_00{k}.png', reconstructed_image)