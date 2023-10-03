import numpy as np

w1 = 0.7
w2 = 1 - w1

T = 10

theta = 1
epsilon = 2
zeta = 3

k = 1000


### FOR NER EXP ###
root_lovelace = "/LUSTRE/users/rnaveiro/corrupting_hmms"
root = "/home/roi.naveiro/corrupting_hmms"
data_path = f'{root_lovelace}/data/ner.csv'
results_path = f'{root_lovelace}/results/'

attack1 = {
    0  : 0,
    1  : 1,
    2  : 1,
    3  : 3,
    4  : 4,
    5  : 5,
    6  : 6,
    7  : 7,
    8  : 8,
    9  : 9,
    10 : 10,
    11 : 11,
    12 : 12,
    13 : 16,
    14 : 16,
    15 : 15,
    16 : 16,
    17 : 17,
    18 : 18,
    19 : 19,
    20 : 20,
    21 : 21,
    22 : 22,
    23 : 23,
    24 : 24,
    25 : 25,
    26 : 26,
    27 : 27,
    28 : 2 
}

attack2 = {
    0  : 13,
    0  : 13,
    0  : 13,
    0  : 13,
    0  : 13,
    0  : 13,
    0  : 13,
    0  : 13,
    1  : 13,
    2  : 13,
    3  : 13,
    4  : 13,
    5  : 13,
    6  : 13,
    7  : 13,
    8  : 13,
    9  : 13,
    10 : 13,
    11 : 13,
    12 : 13,
    13 : 13,
    14 : 13,
    15 : 13,
    16 : 13,
    17 : 13,
    18 : 13,
    19 : 13,
    20 : 13,
    21 : 13,
    22 : 13,
    23 : 13,
    24 : 13,
    25 : 13,
    26 : 13,
    27 : 13,
    28 : 13 
}

