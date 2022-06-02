# %%
import cv2 as cv
import numpy as np
import os
import random
import json
import math
import tqdm
from collections import defaultdict

# %%
base_path = "./GroceryStoreDataset/dataset"

# Read in list of training samples with their class ids
with open(os.path.join(base_path, 'train.txt')) as f:
    train_files = [f.rstrip().split(',') for f in f.readlines()]
    train_files = [(f[0], int(f[1])) for f in train_files]

with open(os.path.join(base_path, 'val.txt')) as f:
    val_files = [f.rstrip().split(',') for f in f.readlines()]
    val_files = [(f[0], int(f[1])) for f in val_files]

with open(os.path.join(base_path, 'test.txt')) as f:
    test_files = [f.rstrip().split(',') for f in f.readlines()]
    test_files = [(f[0], int(f[1])) for f in test_files]

combined_files = train_files + val_files + test_files
print(f"There are {len(combined_files)} samples in total.")

# %%
samples_per_class = 100

class_samples = defaultdict(list)
for sample in combined_files:
    class_samples[sample[1]].append(sample)

used_files = []
for k in list(class_samples.keys()):
    used_files += random.choices(class_samples[k], k=samples_per_class)

print(len(used_files))
print(used_files[0])

# %%
def load_image(path, size=None):
    img = cv.imread(os.path.join(base_path, path))
    if img is not None:        
        if size:
            img = cv.resize(img, size)

        return img, img.shape[0], img.shape[1]
    
    return img, -1, -1

def too_close(new_center, center_list, min_distance):
    for c in center_list:
        if abs(c[0] - new_center[0]) < min_distance and abs(c[1] - new_center[1]) < min_distance:
            return True
    
    return False

def get_balanced_centers(max_per_mosaic, min_distance, img_size):
    num_tiles = random.randint(1, max_per_mosaic)
    centers = [(0,0)] * num_tiles

    for i in range(num_tiles):
        center_i = (random.randint(0, img_size-min_distance), random.randint(0, img_size-min_distance))
        
        # Check that all of our image centers are reasonably distributed and nothing is exremely occluded
        while too_close(center_i, centers, min_distance):
            center_i = (random.randint(0, img_size-min_distance), random.randint(0, img_size-min_distance))

        centers[i] = center_i
    
    return centers

def n_mosaic(index, files, max_per_mosaic, min_distance, img_size):
    centers = get_balanced_centers(max_per_mosaic, min_distance, img_size)
    output_img = np.full((img_size, img_size, 3), 114, dtype=np.uint8)
    all_indices = [index] + [random.randint(0, len(files)-1) for _ in range(len(centers)-1)]
    random.shuffle(all_indices)

    output_labels = [0] * len(centers)
    areas = []

    for i, index in enumerate(all_indices):
        # Load image
        size_scalar = random.random()*2 + 1.5
        img, h, w = load_image(files[index][0], (int(img_size/size_scalar), int(img_size/size_scalar)))
        
        w_rem = int(w % 2)
        h_rem = int(h % 2)

        x1 = centers[i][0] - (w//2)
        x2 = centers[i][0] + (w//2) + w_rem
        y1 = centers[i][1] - (h//2)
        y2 = centers[i][1] + (h//2) + h_rem

        patch_x1 = abs(min(0, x1))
        patch_y1 = abs(min(0, y1))
        patch_x2 = img_size - x1
        patch_y2 = img_size - y1

        x1 = max(0,x1)
        x2 = min(img_size, x2)
        y1 = max(0,y1)
        y2 = min(img_size, y2)


        output_img[y1:y2, x1:x2] = img[patch_y1:patch_y2, patch_x1:patch_x2]
        areas.append((patch_y2 - patch_y1) * (patch_x2 - patch_x1))
        output_labels[i] = (files[index][1])
    
    return output_img, list(set(output_labels)), areas
    
    

#%%
# Can tile UP TO 8 different images
N = 4 # Generate 5 mosaics that are guaranteed to have this image in them
max_per_mosaic = 5
img_size = 512
min_distance = 125 # A minimum distance between image centers

label_dict = {}
label_out_fn = "mosaic_labels.json"
output_dir = "mosaic"
indices = list(range(len(used_files) * N))
train_indices = set(random.sample(indices, int(len(indices) * 0.85)))

os.makedirs(os.path.join(base_path, output_dir), exist_ok=True)
for i, sample in tqdm.tqdm(enumerate(used_files), desc=f'Writing {len(used_files) * N} mosaic ingredient images'):
    for j in range(N):
        mosaic, labels, areas = n_mosaic(i, used_files, max_per_mosaic, min_distance, img_size)
        fn = f"{i}_{j}_{','.join([str(s) for s in labels])}.jpg"
        label_dict[fn] = {'labels': list(set(labels)), 'split': 'train' if (i * N) + j in train_indices else 'val'}
        label_dict[fn]['areas'] = areas
        cv.imwrite(os.path.join(base_path, output_dir, fn), mosaic)

with open(os.path.join(base_path, output_dir, label_out_fn), 'w') as f:
    json.dump(label_dict, f, indent=2)

# %%