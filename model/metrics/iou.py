from torch import zeros

def IoU_torch(preds, targets, num_classes):
    preds = preds.argmax(dim=1)
    intersection = zeros(num_classes)
    union = zeros(num_classes)

    for i in range(num_classes):
        pred_mask = preds == i
        target_mask = targets == i

        intersection[i] = (pred_mask & target_mask).sum().float()
        union[i] = (pred_mask | target_mask).sum().float()

    iou = intersection / (union + 1e-6)

    return iou[0].item(), iou[1].item()