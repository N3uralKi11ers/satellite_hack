from dataset import CVDataset
# from deeplabv3 import DeepLabV3
# from unet import UNet
from unetplusplus import UNetPlusPlus
from torch import no_grad, sigmoid

data = CVDataset('/home/denis/code/PatternCV/', (256,256))

# model = DeepLabV3(2)
# model = UNet(3, 2)
model = UNetPlusPlus(2, 3, deep_supervision=True)

model.eval()
with no_grad():
    index, images, target = data[0]
    images = images.unsqueeze(0)
    output = model(images)[0][-1]
    output = sigmoid(output)
    data.imshow(output)
