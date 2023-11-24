import os
from pathlib import Path

#Вычисляет размер папки, количество файлов и количество итераций функции
def folderSize(path):
    fsize = 0
    numfile = 0
    iteration = 0
    for file in Path(path).rglob('*'):
        if (os.path.isfile(file)):
            fsize += os.path.getsize(file)
            numfile += 1
        iteration += 1
    return fsize, numfile, iteration
  
  
folder = './train_splitted' # train fragments dir path

print("Вычисление размера выбранной папки...")
size, numfile, iteration = folderSize(folder)
print(f'Выбрана папка: {folder}')
print(f'Найдено файлов: {numfile}')
print("Размер папки:")
print(f'{size} Bytes')       
print(f'{size/1048576:.2f} Mb')
print(f'{size/1073741824:.2f} Gb')