import os
import shutil

root_dir = r'D:\hqy\datasets\IQA\LIVE2005(pre)\distorted_images'
datasets = ['fastfading', 'gblur', 'jp2k', 'jpeg', 'wn']

for dataset in datasets:
    if os.path.exists(os.path.join(root_dir, dataset)) == False:
        os.mkdir(os.path.join(root_dir, dataset))

for name in os.listdir(root_dir):
    if name.find('.') == -1:
        continue

    original_name = name
    name, dir = name.split('_')
    dir, surname = dir.split('.')
    name = name + '.' + surname
    print(name, ' ', dir)
    shutil.move(os.path.join(root_dir, original_name), os.path.join(root_dir, dir, name))