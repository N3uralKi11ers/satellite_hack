import torch

def f1_scores(y_true, y_pred, threshold=0.5):
    y_pred = y_pred > threshold

    f1_scores = []

    for i in range(y_pred.shape[1]):
        y_pred_class = y_pred[:, i]
        
        tp = torch.sum(y_true * y_pred_class).float()
        fp = torch.sum((1 - y_true) * y_pred_class).float()
        fn = torch.sum(y_true * torch.logical_not(y_pred_class)).float()
        
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        f1_scores.append(f1.item())
    
    return f1_scores[0], f1_scores[1]