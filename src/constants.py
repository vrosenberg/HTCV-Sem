import os

MAP_PERCENT_TO_AMOUNT = {
    "10" : 280,
    "25" : 700,
    "50" : 1400,
    "100": 2799
}

DAMAGE_LEVEL_TO_SCORE = {
    "destroyed" : 4,
    "major-damage" : 3,
    "minor-damage" : 2,
    "no-damage" : 1,
}

DATASET_FOLDER_PATH = os.path.join(os.path.dirname(__file__),"..","data")
PREDETERMINED_RANDOM_DATA_PATH = os.path.join(DATASET_FOLDER_PATH,"xview2.txt")
