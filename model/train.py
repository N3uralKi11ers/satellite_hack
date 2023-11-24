from deeplabv3 import DeepLabV3
from unet import UNet
from unetplusplus import UNetPlusPlus
from segnet import SegNet
from dataset import CVDataset
from metrics import IoU_torch, f1_scores
from inference_dataset import CVInferenceDataset

from torch.utils.data import DataLoader
from os import cpu_count
from torch import no_grad
from torch.cuda import is_available as cuda_is_available
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from torch import tensor, cuda, is_tensor
import matplotlib.pyplot as plt
import gc
import os
import shutil

device = 'cuda' if cuda_is_available() else 'cpu'
num_epochs = 10
batch_size = 5

data_train = CVInferenceDataset('/home/denis/code/satellite_hack/', 3, True, (256,256))
data_valid = CVInferenceDataset('/home/denis/code/satellite_hack/', 3, False, (256,256))
loader = DataLoader(data_train, batch_size=batch_size, shuffle=False, num_workers=cpu_count(), pin_memory=False)
weight_loss = [1., 1., 1., 1.]

def visual_graphic(list_res_model):
    '''
    model:
        epoch:
            k:
                k, train_loss, val_loss, val_two_score, val_one_score
    '''
    '''
    epoch1:
        model:
            k...
    epoch2:
    ...
    '''
    model_names = list(list_res_model.keys())
    model_values = list(list_res_model.values())
    for i in range(len(model_values[0])):
        '''
        Создаем фигуру для определенной эпохи и рисуем для каждой модели ее скор в определенную итерацию, рисуем график, сохраняем
        '''
        if os.path.exists(f"/home/denis/code/satellite_hack/model/graphic_res/{i+1}epochs"): 
            shutil.rmtree(f"/home/denis/code/satellite_hack/model/graphic_res/{i+1}epochs")
        os.makedirs(f"/home/denis/code/satellite_hack/model/graphic_res/{i+1}epochs")
        k = model_values[0][i][0]
        plt.figure(figsize=(16, 9))
        plt.title('Loss')
        for j in range(len(model_names)):
            y1 = model_values[j][i][1]
            y2 = model_values[j][i][2]
            plt.plot(k, y1, label=f'train {model_names[j]}')
            plt.plot(k, y2, label=f'val {model_names[j]}')
            plt.legend(bbox_to_anchor=(0.98, 1), loc='upper left')
        plt.savefig(f"/home/denis/code/satellite_hack/model/graphic_res/{i+1}epochs/loss.png")
        plt.figure(figsize=(16, 9))
        plt.title("Score")
        for j in range(len(model_names)):
            y1 = model_values[j][i][4]
            y2 = model_values[j][i][3]
            plt.plot(k, y1, label=f'build {model_names[j]}')
            plt.plot(k, y2, label=f'back {model_names[j]}')
            plt.legend(bbox_to_anchor=(0.98, 1), loc='upper left')
        plt.savefig(f"/home/denis/code/satellite_hack/model/graphic_res/{i+1}epochs/score.png")

def calculate_valid(model_names, model, data, criterion, score):
    val_los = 0
    val_score_mean = 0
    val_score_one = 0
    with no_grad():
        for j in range(len(data)):
            _, images, targets = data[j]
            images = images.unsqueeze(0).to(device)
            targets = targets.unsqueeze(0).to(device)
            output = model(images)
            if model_names != 'unetSV++':
                loss = criterion(output, targets)
            else:
                loss = criterion[0](output[0], targets) * weight_loss[0]
                for i in range(1, 4):
                    loss += criterion[i](output[i], targets) * weight_loss[i]
                output = output[-1]
            output = output.detach().cpu()
            data.imshow(images[0].detach().cpu(), targets[0].detach().cpu(), output[0].detach().cpu(), f'test{j}')
            val_los += loss.item()
            val_score_mn, val_score_on = score(output, targets.detach().cpu())
            val_score_mean += val_score_mn
            val_score_one += val_score_on
    return val_los / batch_size, val_score_mean / batch_size, val_score_one / batch_size

def clear_cuda_memory():
    cuda.empty_cache()

    for obj in gc.get_objects():
        try:
            if is_tensor(obj) and obj.is_cuda:
                del obj
        except Exception as e:
            pass

    gc.collect()
    cuda.empty_cache()


list_model = {'unet': UNet(3, 2),
              'unetSV++': UNetPlusPlus(2, 3, deep_supervision=True),
              'unet++': UNetPlusPlus(2, 3, deep_supervision=False),
              'deeplab': DeepLabV3(2)
              }
list_res_model = {}

for model_names in list_model.keys():
    model = list_model[model_names].to(device)
    criterion = [CrossEntropyLoss(weight=tensor([0.1, 1.], device=device),reduction='mean') for _ in range(4)]
    if model_names != 'unetSV++':
        criterion = criterion[0]
    optim = Adam(model.parameters(), lr=3e-4)
    score = f1_scores
    model.train()
    epoch_result = []
    for _ in tqdm(range(num_epochs)):
        graphic_val_loss = []
        graphic_val_two_accuracy = []
        graphic_val_one_accuracy = []
        graphic_train_loss = []
        list_k = []

        for k, (index, images, targets) in enumerate(tqdm(loader)):
            images, targets = images.to(device), targets.to(device)
            optim.zero_grad()
            output = model(images)
            if model_names != 'unetSV++':
                loss = criterion(output, targets)
            else:
                loss = criterion[0](output[0], targets) * weight_loss[0]
                for i in range(1, 4):
                    loss += criterion[i](output[i], targets) * weight_loss[i]
            loss.backward()
            optim.step()
            if k % 1 == 0:
                model.eval()
                val_los, val_score_one, val_score_two = calculate_valid(model_names, model, data_valid, criterion, score)
                model.train()
                list_k.append(k)
                graphic_train_loss.append(loss.item())
                graphic_val_loss.append(val_los)
                graphic_val_two_accuracy.append(val_score_two)
                graphic_val_one_accuracy.append(val_score_one)
        epoch_result.append((list_k, graphic_train_loss, graphic_val_loss, graphic_val_two_accuracy, graphic_val_one_accuracy))
    del model, criterion, optim, score
    clear_cuda_memory()
    list_res_model[model_names] = epoch_result

visual_graphic(list_res_model)