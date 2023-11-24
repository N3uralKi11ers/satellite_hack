import pandas as pd

data = pd.DataFrame({'image_id': [], 'split': [], 'sat_image_path': [], 'mask_path': []})

for i in range(21):
    j = i
    if j < 10:
        j = f'0{j}'
    data.loc[i] = [i, 'train', f'train_updated_titiles/images/train_image_0{j}.png', f'train_updated_titiles/masks/train_mask_0{j}.png']

data.to_csv('/home/denis/code/satellite_hack/metadata.csv', index=False)