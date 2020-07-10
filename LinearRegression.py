import matplotlib.pyplot as plt
import numpy as np
from pylab import *


with open('student-mat.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Columns: {", ".join(row)}')
            line_count += 1
print(f'{line_count} lines.')





