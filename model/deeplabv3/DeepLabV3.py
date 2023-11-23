from torch.nn import Module, Conv2d
from torchvision.models.segmentation import deeplabv3_resnet101

class DeepLabV3(Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = deeplabv3_resnet101(pretrained=False, weights='DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1')
        self.model.classifier = Conv2d(2048, num_classes, kernel_size=(1, 1))

    def forward(self, input):
        output = self.model(input)['out']
        return output
