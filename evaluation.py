import evaluation_lib
import numpy as np
import code


## Create numpy array for different thresholds
threshold_list = np.arange(1, 101)/100
print("Evaluating the following thresholds:", threshold_list)


for threshold in np.nditer(threshold_list):
    evaluation_lib.evaluate(threshold)


code.interact(local=dict(globals(), **locals()))
