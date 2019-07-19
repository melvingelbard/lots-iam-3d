import numpy as np
import csv
import sys
import os, pathlib

with open('input.csv', mode='w') as input_csv:
    input_csv = csv.writer(input_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator = '\n')
    for i in range(1, 61):
        for j in range(1, 4):
            if i < 10:
                i = "0" + str(i)
            input_csv.writerow(['D:/Edinburgh/dissertation/data', 'DMP' + str(i), 'V' + str(j)])
