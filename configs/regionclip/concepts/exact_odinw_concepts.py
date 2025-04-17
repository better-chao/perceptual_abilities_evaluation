import os
import json

data_dir = "/gpfsdata/home/huangziyue/data/odinw"

subdirs = [
    "AerialMaritimeDrone/large/test",
    "AerialMaritimeDrone/tiled/test",
    "AmericanSignLanguageLetters/American Sign Language Letters.v1-v1.coco/test",
    "Aquarium/Aquarium Combined.v2-raw-1024.coco/test",
    "BCCD/BCCD.v3-raw.coco/test",
    "boggleBoards/416x416AutoOrient/export",
    "brackishUnderwater/960x540/test",
    "ChessPieces/Chess Pieces.v23-raw.coco/test",
    "CottontailRabbits/test",
    "dice/mediumColor/export",
    "DroneControl/Drone Control.v3-raw.coco/test",
    "EgoHands/generic/test",
    "EgoHands/specific/test",
    "HardHatWorkers/raw/test",
    "MaskWearing/raw/test",
    "MountainDewCommercial/test",
    "NorthAmericaMushrooms/North American Mushrooms.v1-416x416.coco/test",
    "openPoetryVision/512x512/test",
    "OxfordPets/by-breed/test",
    "OxfordPets/by-species/test",
    "Packages/Raw/test",
    "PascalVOC/valid",
    "pistols/export",
    "PKLot/640/test",
    "plantdoc/416x416/test",
    "pothole/test",
    "Raccoon/Raccoon.v2-raw.coco/test",
    "selfdrivingCar/fixedLarge/export",
    "ShellfishOpenImages/raw/test",
    "ThermalCheetah/test",
    "thermalDogsAndPeople/test",
    "UnoCards/raw/test",
    "VehiclesOpenImages/416x416/test",
    "websiteScreenshots/test",
    "WildfireSmoke/test",
    ]

subdir_keys = [
    'AerialMaritimeDrone_large',
    'AerialMaritimeDrone_tiled',
    'AmericanSignLanguageLetters',
    'Aquarium',
    'BCCD',
    'boggleBoards',
    'brackishUnderwater',
    'ChessPieces',
    'CottontailRabbits',
    'dice_mediumColor',
    'DroneControl',
    'EgoHands_generic',
    'EgoHands_specific',
    'HardHatWorkers',
    'MaskWearing',
    'MountainDewCommercial',
    'NorthAmericaMushrooms',
    'openPoetryVision',
    'OxfordPets_by-breed',
    'OxfordPets_by-species',
    'Packages',
    'PascalVOC',
    'pistols',
    'PKLot',
    'plantdoc',
    'pothole',
    'Raccoon',
    'selfdrivingCar',
    'ShellfishOpenImages',
    'ThermalCheetah',
    'thermalDogsAndPeople',
    'UnoCards',
    'VehiclesOpenImages',
    'websiteScreenshots',
    'WildfireSmoke',
]

image_dirs = [os.path.join(data_dir, subdir) for subdir in subdirs]
# print(image_dirs)

anno_file_name = "annotations_without_background.json"

anno_files = [os.path.join(img_dir, anno_file_name) for img_dir in image_dirs]
# print(anno_files)

new_anno_files = []
for anno_file in anno_files:
    if not os.path.exists(anno_file):
        img_dir = os.path.dirname(anno_file)
        # print(img_dir)
        new_anno_file_name = "test_annotations_without_background.json"
        anno_file = os.path.join(img_dir, new_anno_file_name)
    new_anno_files.append(anno_file)
# print(new_anno_files)

# for anno_file in new_anno_files:
#     if not os.path.exists(anno_file):
#         print(f"{anno_file} not exists!!!")

# print(len(image_dirs))
# print(len(new_anno_files))

# 这里为了生成新的概念
concepts_dir = '/gpfsdata/home/yangshuai/open_vocabulary/RegionCLIP/concepts/odinw'

concepts_sub_dir = [os.path.join(concepts_dir, subkey) for subkey in subdir_keys]
# print(concepts_sub_dir)

for i in range(35):
    os.makedirs(concepts_sub_dir[i], exist_ok = True)

    with open(new_anno_files[i], 'r') as ann_file:
        data = json.load(ann_file)

    print(data['categories'])
    
    categories = data['categories']

    concepts_file_path = os.path.join(concepts_sub_dir[i], 'concepts.txt')

    with open(concepts_file_path, 'w') as save_file:
        for item in categories:
            class_name = item['name']
            save_file.write(class_name + '\n')


# 这里是为了注册 odinw 时用到的，写在builtin.py
# print(len("/gpfsdata/home/huangziyue/data/"))
# for i in range(35):
#     print(f"\"{subdir_keys[i]}\": (\"{image_dirs[i][31:]}\", \"{new_anno_files[i][31:]}\"),")