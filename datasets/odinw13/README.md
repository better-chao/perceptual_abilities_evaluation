# GLIP: Grounded Language-Image Pre-training

[GLIP: Grounded Language-Image Pre-training](https://github.com/microsoft/GLIP)

## Abstract

ODinW was first proposed in GLIP and refined and formalized in ELEVATER. 
GLIP used 13 downstream tasks while the full ODinW has 35 downstream tasks.

## Dataset Path and Annotation Explanation
```shell
/gpfsdata/home/huangziyue/data/odinw
|---AerialMaritimeDrone
|   |---xxx
|   |---xxx
|   |---xxx
|   |...
|---AmericanSignLanguageLetters
|   |---xxx
|   |---xxx
|   |---xxx
|   |...
|---Aquarium
|   |---xxx
|   |---xxx
|   |---xxx
|   |...
|...
```
Odinw13 and Odinw35 use the same folder, with each folder 
representing a sub-dataset. Odinw13 contains 13 sub-datasets, 
while Odinw35 includes all 35.

## YAML Config Explanation
```shell
odinw13/
|---AerialMaritimeDrone_large.yaml
|---Aquarium_Aquarium_Combined.v2-raw-1024.coco.yaml
|---CottontailRabbits.yaml
|...
``` 
Each file is a configuration file for a sub-dataset.
Introduction of Some Keywords:

OVERRIDE_CATEGORY: Category dictionary.
TRAIN: Training set.
TEST: Test set.
INPUT: Image size.

















