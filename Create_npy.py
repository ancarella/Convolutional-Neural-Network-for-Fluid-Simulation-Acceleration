
import numpy as np
import openpyxl as xl
import pandas as pd
import os
import pickle

i = 0

velocity_vector = (0.1125, 0.3125, 0.5125)

batch = 0

for v in velocity_vector:

    folder = "./Square/Val/CSVs/" + str(v) + "/"
    path, dirs, files = next(os.walk(folder))
    file_count = len(files)
    print(file_count)
    print(str(int(file_count / 3)))
    a = np.zeros(shape=(int(file_count / 3), 3, 66, 256))

    print(str(v) + "   " + folder)
    for file_name in os.listdir(folder):
        data_number = int(file_name[0:6])
        if file_name[-5] == "e":
            a[data_number][0] = pd.DataFrame(pd.read_csv(folder + file_name)).to_numpy()  # pressure

        elif file_name[-5] == "X":
            a[data_number][1] = pd.DataFrame(pd.read_csv(folder + file_name)).to_numpy()  # Velocity X

        else:
            a[data_number][2] = pd.DataFrame(pd.read_csv(folder + file_name)).to_numpy()  # Velocity Y

        i = i + 1
    np.save("./Square/Val/NPYs/npy " + str(v), a)
