import numpy as np
import random
r1_path = "/home/like/data/trajectory_similarity/beijing/beijing100200/exp_data_r1.npy"
r1_path = "/home/like/data/trajectory_similarity/porto/porto100300/exp_data_r1.npy"
r1_data = np.load(r1_path, allow_pickle=True)
r2_path = "/home/like/data/trajectory_similarity/beijing/beijing100200/exp_data_r2.npy"
r2_path = "/home/like/data/trajectory_similarity/porto/porto100300/exp_data_r2.npy"
r2_data = np.load(r2_path, allow_pickle=True)
trj_id = random.randint(0, len(r1_data[0]) if len(r1_data[0]) < len(r2_data[0]) else len(r2_data[0]))
for each_r in r1_data:
    print(len(each_r[trj_id][0]))
for each_r in r2_data:
    print(len(each_r[trj_id][0]))