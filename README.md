# HTCV-Sem
Baseline and modified model of xView2 challenge for a Computer Vision Seminar

USING OPENCV 4.7.0

# Augmented Data Metrics

classes
0 , 1 , 2 , 3 , 4
## base_10 
randomly chosen from the base 10%

new
[714 ,1134, 501, 422, 230]

old
[663, 1172, 506, 420, 239]

## base_25
randomly chosen from the base 25%

new
[1669, 2795, 1047, 877, 612]

old
[1568, 2845, 1044, 908, 635]

## base_50
randomly chosen from the base 50%

new
[2930, 5675, 2207, 2025, 1163]

old
[2767, 5756, 2202, 2098, 1177]

## base_100
randomly chosen from the base 100%

new:
[5971, 11324, 4343, 3992, 2370]

old:
[5566, 11433, 4357, 4230, 2414]

Overall Accuracy: 0.603
Average Accuracy: 0.491

## only_all
each image from original 100 got augmented
new
[601, 1126, 459, 382, 231]

old
[558, 1151, 437, 410, 243]

## no_resize_25
randomly chosen from the base 10% without resize

class instances:
[718, 1150, 481, 396, 255]

class_instances_orig:
[710, 1155, 480, 395, 260]

performance:

10epoch:
Overall Accuracy: 0.565
Average Accuracy: 0.386

50epoch:
Overall Accuracy: 0.563
Average Accuracy: 0.477

## resize_25
class instances:
[679, 1237, 463, 369, 252]

class_instances_orig:
[643, 1264, 451, 387, 255]

performance:

10epoch:
Overall Accuracy: 0.535
Average Accuracy: 0.363

50epoch
Overall Accuracy: 0.595
Average Accuracy: 0.483

# Baseline Metrics

## 100% Dataset
50 epoch
Overall Accuracy: 0.623
Average Accuracy: 0.520

## 50% Dataset
50 epoch
Overall Accuracy: 0.603
Average Accuracy: 0.503

## 25% Dataset
50 epoch
Overall Accuracy: 0.582
Average Accuracy: 0.462

## 10% Dataset
50 epoch
Overall Accuracy: 0.538
Average Accuracy: 0.352

## base_10 1 aug to 2 orig
orig:
{'0': 58, '1': 115, '2': 43, '3': 40, '4': 24}
aug:
{'0': 29, '1': 58, '2': 22, '3': 20, '4': 12}

## base_10 2 aug to 1 orig
orig:
{'0': 58, '1': 115, '2': 43, '3': 40, '4': 24}
aug:
{'0': 116, '1': 230, '2': 86, '3': 80, '4': 48}

## base_10 1 aug to 1 orig
orig:
{'0': 58, '1': 115, '2': 43, '3': 40, '4': 24}
aug:
{'0': 58, '1': 115, '2': 43, '3': 40, '4': 24}

## base_25 2 aug to 1 orig
orig
{'0': 155, '1': 285, '2': 109, '3': 90, '4': 61}
aug
{'0': 310, '1': 570, '2': 218, '3': 180, '4': 122}
## base_25 1 aug to 1 orig
orig
{'0': 155, '1': 285, '2': 109, '3': 90, '4': 61}
aug
{'0': 155, '1': 285, '2': 109, '3': 90, '4': 61}

## base_25 1 aug to 2 orig
orig
{'0': 155, '1': 285, '2': 109, '3': 90, '4': 61}
aug
{'0': 78, '1': 142, '2': 54, '3': 45, '4': 30}

## base_50 2 aug to 1 orig
orig
{'0': 288, '1': 571, '2': 216, '3': 209, '4': 116}
aug
{'0': 576, '1': 1142, '2': 432, '3': 418, '4': 232}

## base_50 1 aug to 1 orig
orig
{'0': 288, '1': 571, '2': 216, '3': 209, '4': 116}
aug
{'0': 288, '1': 571, '2': 216, '3': 209, '4': 116}

## base_50 1 aug to 2 orig
orig
{'0': 288, '1': 571, '2': 216, '3': 209, '4': 116}
aug
{'0': 144, '1': 286, '2': 108, '3': 104, '4': 58}

## base_100 1 aug to 2 orig
orig
{'0': 558, '1': 1151, '2': 437, '3': 410, '4': 243}
aug
{'0': 279, '1': 576, '2': 218, '3': 205, '4': 122}
## base_100 1 aug to 1 orig
orig
{'0': 558, '1': 1151, '2': 437, '3': 410, '4': 243}
aug
{'0': 558, '1': 1151, '2': 437, '3': 410, '4': 243}
## base_100 2 aug to 1 orig
orig
{'0': 558, '1': 1151, '2': 437, '3': 410, '4': 243}
aug
{'0': 1116, '1': 2302, '2': 874, '3': 820, '4': 486}