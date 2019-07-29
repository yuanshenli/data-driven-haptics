
import numpy as np

file_name = "7_24_15_55.txt"

x = []

force_thresh = 1024.0

with open(file_name) as f:
	for idx, line in enumerate(f):
		line = line.split(",")
		d0 = float(line[0])
		d1 = float(line[1])
		d2 = float(line[2])
		if d1 > force_thresh:
			d1 = force_thresh
		x.append([d0, d1, d2])

if f:
	f.close()

x = np.asarray(x)
print(x.max(axis=0))

x_normed = (x - x.min(axis=0)) / x.ptp(axis=0)
print(np.average(x_normed,axis=0))

normalized_file_name = "7_24_15_55_norm.txt"
np.savetxt(normalized_file_name, x_normed, delimiter=',')


