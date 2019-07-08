import numpy as np
import code
import pptk


combined_age_map = np.load("D:\edinburgh\dissertation\combined_age_map_mri_mult_normed.npy")
mri_data = np.load("D:\edinburgh\dissertation\mri_data.npy")
thrsh_damaged = combined_age_map
thrsh_sane = np.nan_to_num(mri_data)


damaged = []
thrsh_damaged[thrsh_damaged<0.5] = 0

thrsh_sane[thrsh_sane>0.5] = 0

damaged = np.argwhere(thrsh_damaged)
sane = np.argwhere(thrsh_sane)
rgb_damaged = np.zeros(damaged.shape)
rgb_sane = np.zeros(sane.shape)



for point in range(damaged.shape[0]):
    coordinates = damaged[point]
    rgb_damaged[point] = round(thrsh_damaged[coordinates[0], coordinates[1], coordinates[2]]*128+128)


for point in range(sane.shape[0]):
    coordinates = sane[point]
    rgb_sane[point] = round(thrsh_sane[coordinates[0], coordinates[1], coordinates[2]]*80)

code.interact(local=dict(globals(), **locals()))
scalars = rgb_damaged[:, 0]
scalars = np.concatenate((scalars, rgb_sane[:, 0]))

code.interact(local=dict(globals(), **locals()))


whole_brain = np.concatenate((damaged, sane))

v = pptk.viewer(whole_brain, scalars)
v.color_map('jet', scale=[0, 255])
v.set(point_size=0.3)




code.interact(local=dict(globals(), **locals()))
