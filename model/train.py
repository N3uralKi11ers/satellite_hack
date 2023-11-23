from deeplabv3 import DeepLabV3
from unet import UNet
from unetplusplus import UNetPlusPlus
from segnet import SegNet
from dataset import CVDataset
from metrics import IoU_torch

from torch.utils.data import DataLoader
from os import cpu_count
from torch import no_grad
from torch.cuda import is_available as cuda_is_available
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from torch import tensor
import matplotlib.pyplot as plt

device = 'cuda' if cuda_is_available() else 'cpu'
num_epochs = 6
batch_size = 5

data = CVDataset('/home/denis/code/PatternCV/', matrix_shape=(256,256))
loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=cpu_count())

# model = UNet(3, 2).to(device)
model = UNetPlusPlus(2, 3, deep_supervision=True).to(device)
# model = DeepLabV3(2).to(device)
# model = SegNet(3, 2).to(device)

# criterion = CrossEntropyLoss(weight=tensor([10., 1.], device=device),reduction='mean')
criterion = [CrossEntropyLoss(weight=tensor([10., 1.], device=device),reduction='sum') for _ in range(4)]
weight_loss = [1., 1., 1., 3.]
optim = Adam(model.parameters(), lr=3e-4)
score = IoU_torch

def visual_graphic(list_k, graphic_train_loss, graphic_val_loss, graphic_val_mean_accuracy, graphic_val_one_accuracy):
    pass

def calculate_valid():
    val_los = 0
    val_score_mean = 0
    val_score_one = 0
    with no_grad():
        for j in range(batch_size):
            _, images, targets = data[j]
            images = images.unsqueeze(0).to(device)
            targets = targets.unsqueeze(0).to(device)
            output = model(images)
            # los = criterion(output, targets)
            los = criterion[0](output[0], targets) * weight_loss[0]
            for i in range(1, 4):
                los += criterion[i](output[i], targets) * weight_loss[i]
            output = output[-1].detach().cpu()
            data.imshow(images[0].detach().cpu(), targets[0].detach().cpu(), output[0].detach().cpu(), f'test{j}')
            val_los += los.item()
            val_score_mn, val_score_on = score(output, targets.detach().cpu(), data.class_count)
            val_score_mean += val_score_mn
            val_score_one += val_score_on
    return val_los / batch_size, val_score_mean / batch_size, val_score_one / batch_size


model.train()
for _ in tqdm(range(num_epochs)):
    graphic_val_loss = []
    graphic_val_mean_accuracy = []
    graphic_val_one_accuracy = []
    graphic_train_loss = []
    list_k = []

    for k, (index, images, targets) in enumerate(tqdm(loader)):
        images, targets = images.to(device), targets.to(device)
        optim.zero_grad()
        output = model(images)
        # loss = criterion(output, targets)
        loss = criterion[0](output[0], targets) * weight_loss[0]
        for i in range(1, 4):
            loss += criterion[i](output[i], targets) * weight_loss[i]
        loss.backward()
        optim.step()
        if k % 20 == 0:
            model.eval()
            val_los, val_score_mean, val_score_one = calculate_valid()
            print(f'\nTrain Loss: {loss.item()}, Val Loss: {val_los}\nVal Mean Score: {val_score_mean}, Val Score 1cl: {val_score_one}')
            model.train()
            list_k.append(k)
            graphic_train_loss.append(loss.item())
            graphic_val_loss.append(val_los)
            graphic_val_mean_accuracy.append(val_score_mean)
            graphic_val_one_accuracy.append(val_score_one)
    # visual_graphic(list_k, graphic_train_loss, graphic_val_loss, graphic_val_mean_accuracy, graphic_val_one_accuracy)