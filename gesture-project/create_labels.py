import numpy as np

labels = np.array(['l', 'peace', 'stop', 'thumbs_up'])
np.save("model/class_labels.npy", labels)

print("class_labels.npy created!")