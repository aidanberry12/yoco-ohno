import numpy as np
import pickle
import cv2 as cv
import numpy as np
import os
import random
import json
import math
import tqdm
from collections import defaultdict

base_path = "./GroceryStoreDataset/dataset"

print('reading train, test, and val images')
with open(os.path.join(base_path, 'train.txt')) as f:
        train_files = [f.rstrip().split(',') for f in f.readlines()]
        train_files = [(f[0], int(f[2])) for f in train_files]

with open(os.path.join(base_path, 'val.txt')) as f:
    val_files = [f.rstrip().split(',') for f in f.readlines()]
    val_files = [(f[0], int(f[2])) for f in val_files]

with open(os.path.join(base_path, 'test.txt')) as f:
    test_files = [f.rstrip().split(',') for f in f.readlines()]
    test_files = [(f[0], int(f[2])) for f in test_files]

print('generating class samples dict')
combined_files = train_files + val_files + test_files
class_samples = defaultdict(list)
for sample in combined_files:
    class_samples[sample[1]].append(sample)

mapping = {'Apple': ['apple', 'applejack', 'pineapple', 'crabapples', 'applesauce', 'fiber_supplement'],
 'Avocado': ['avocado', 'hass_avocadoes'],
 'Banana': ['banana', 'dried_banana_pieces', 'creme_de_banane'],
 'Kiwi': ['kiwi'],
 'Lemon': ['lemon', 'lemonade', 'dried_lemon_grass', 'carbonated_lemon_-_lime_beverage', 'limoncello', 'bacardi_limon'],
 'Lime': ['lime', 'limeade', 'fresh_lime_leaves'],
 'Mango': ['mango', 'orange', 'instant_tang_orange_drink', 'persimmon', 'tangelo', 'tang_orange_crystals', 'frozen_mango_chunks', 'tangerine'],
 'Melon': ['melon', 'watermelon', 'bitter_melons'],
 'Nectarine': ['nectarine', 'tartaric', 'kumquat', 'persimmon', 'tangerine'],
 'Orange': ['orange', 'persimmon', 'tangelo', 'tang_orange_crystals', 'instant_tang_orange_drink', 'tangerine', 'kumquat'],
 'Papaya': ['papaya', 'pawpaw'],
 'Passion-Fruit': ['fruit', 'citrus_fruits', 'citron'],
 'Peach': ['peach', 'cling_peach_halves', 'canned_peach_halves'],
 'Pear': ['pear'],
 'Pineapple': ['pineapple'],
 'Plum': ['plum', 'pluots'],
 'Pomegranate': ['pomegranate'],
 'Red-Grapefruit': ['grapefruit', 'fruit', 'dried_fruits', 'citrus_fruits', 'jackfruit'],
 'Satsumas': ['orange', 'persimmon', 'tangelo', 'tang_orange_crystals', 'instant_tang_orange_drink', 'tangerine', 'kumquat'],
 'Juice': ['juice', 'ice', 'verjuice', 'reserved_juices'],
 'Milk': ['milk', 'soymilk', 'buttermilk'],
 'Oatghurt': ['yoghurt', 'yogurt' 'oats'],
 'Oat-Milk': ['milk', 'soymilk', 'buttermilk', 'oats', 'powdered_chocolate_milk_mix'],
 'Sour-Cream': ['cream', 'dream_whip', 'recipe_cream_filling', 'dairy_creamer', 'sour_mix'],
 'Sour-Milk': ['milk', 'soymilk', 'sour_mix', 'buttermilk'],
 'Soyghurt': ['yoghurt', 'yogurt', 'khoya'],
 'Soy-Milk': ['soymilk', 'milk', 'buttermilk'],
 'Yoghurt': ['yoghurt', 'yogurt'],
 'Asparagus': ['asparagus'],
 'Aubergine': ['eggplant', 'aubergine'],
 'Cabbage': ['cabbage', 'rutabaga'],
 'Carrots': ['carrot', 'sprouts'],
 'Cucumber': ['cucumber', 'thin_cucumber_slices'],
 'Garlic': ['garlic'],
 'Ginger': ['ginger', 'gingerroot'],
 'Leek': ['leek'],
 'Mushroom': ['mushroom'],
 'Onion': ['onion'],
 'Pepper': ['pepper', 'sweet_red_pepper_strips', 'korean_red_pepper_paste'],
 'Potato': ['potato', 'tater_tots', 'au_gratin_potato_mix', 'frozen_potato_slices', 'scalloped_potatoes_mix'],
 'Red-Beet': ['beet', 'beetroot'],
 'Tomato': ['tomato', 'tomatillo', 'roma', 'sun_-_dried_tomato_pesto', 'green_tomatillo_sauce', 'sun_-_dried_tomato_dressing'],
 'Zucchini': ['zucchini']}
flattened_values = [v for r in mapping.values() for v in r]
print('initial setup complete')

# gets the grocery ingredient mapping of all related ingredients in the provided recipe1M list
def map_r1m_to_groc(recipe_1m_ingr_list, groc_mapping, flattened_values):
    groc_list = []
    for r1m_item in recipe_1m_ingr_list:
        if r1m_item in flattened_values:
            for k, v in groc_mapping.items():
                if r1m_item in v:
                    # add the grocery item to the final list
                    groc_list.append(k)
                    break
        else:
            continue
    return groc_list

def find_matching_recipes(recipe_list, recipe_ids, groc_mapping, r1m_overlap_flat, threshold):
    #matched_recipes = []
    grocery_combos = set()
    no_overlapping = set()
    for idx, (recipe, id) in enumerate(zip(recipe_list, recipe_ids)):
        groc_ingrs_set = set(map_r1m_to_groc(recipe, groc_mapping, r1m_overlap_flat))
        match_freq = len(groc_ingrs_set)
        # there are more than threshold grocery ingredients in the recipe1M recipe
        if match_freq >= threshold:
            matched_tuple = (list(groc_ingrs_set), recipe)
            overlap_check = tuple(groc_ingrs_set)
            # if that combination of grocery ingredients has not yet been matched add it to the final list
            if overlap_check not in no_overlapping:
                grocery_combos.add((tuple(groc_ingrs_set), idx, id))
                no_overlapping.add(tuple(groc_ingrs_set))
                
                #matched_recipes.append(matched_tuple)
    return grocery_combos


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

def get_balanced_centers(num_tiles, min_distance, img_size):
    #num_tiles = random.randint(1, max_per_mosaic)
    centers = [(0,0)] * num_tiles

    for i in range(num_tiles):
        center_i = (random.randint(0, img_size-min_distance), random.randint(0, img_size-min_distance))
        
        # Check that all of our image centers are reasonably distributed and nothing is exremely occluded
        while too_close(center_i, centers, min_distance):
            center_i = (random.randint(0, img_size-min_distance), random.randint(0, img_size-min_distance))

        centers[i] = center_i
    
    return centers

def manual_mosaic(indices, files_per_idx,  min_distance, img_size):
    centers = get_balanced_centers(len(indices), min_distance, img_size)
    output_img = np.full((img_size, img_size, 3), 114, dtype=np.uint8)
    mosaic_imgs = []
    for idx in indices:
        sampled_img = random.choices(class_samples[idx], k=1)[0]
        mosaic_imgs.append(sampled_img)
    random.shuffle(mosaic_imgs)

    output_labels = [0] * len(centers)
    areas = []

    for i, img in enumerate(mosaic_imgs):
        # Load image
        img = img[0]
        size_scalar = random.random()*2 + 1.5
        img, h, w = load_image(img, (int(img_size/size_scalar), int(img_size/size_scalar)))
        
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
    return output_img



print('starting mosaic creation')
# load recipes and indices
r1m_recipes_train = pickle.load(open('recipe1m_train.pkl','rb'))
r1m_ingredient_lists_train = [d['ingredients'] for d in r1m_recipes_train]
r1m_id_train = [d['id'] for d in r1m_recipes_train]

r1m_recipes_test = pickle.load(open('recipe1m_test.pkl','rb'))
r1m_ingredient_lists_test = [d['ingredients'] for d in r1m_recipes_test]
r1m_id_test = [d['id'] for d in r1m_recipes_test]

r1m_recipes_val = pickle.load(open('recipe1m_val.pkl','rb'))
r1m_ingredient_lists_val = [d['ingredients'] for d in r1m_recipes_val]
r1m_id_val = [d['id'] for d in r1m_recipes_val]

print('finished reading in recipes')

grocery_combos_2_train = find_matching_recipes(r1m_ingredient_lists_train, r1m_id_train, mapping, flattened_values, threshold = 2)
grocery_combos_2_test = find_matching_recipes(r1m_ingredient_lists_test, r1m_id_test, mapping, flattened_values, threshold = 2)
grocery_combos_2_val = find_matching_recipes(r1m_ingredient_lists_val, r1m_id_val, mapping, flattened_values, threshold = 2)

print('finished mapping grocery to recipe1M')

groc_idx = {k: i for i, (k, v) in enumerate(mapping.items()) }

groc_idx_sets_2_train = []
for combo, r_idx, r_id in grocery_combos_2_train:
    idx_tup = ([groc_idx[item] for item in combo], r_idx, r_id)
    groc_idx_sets_2_train.append(idx_tup)

groc_idx_sets_2_test = []
for combo, r_idx, r_id in grocery_combos_2_test:
    idx_tup = ([groc_idx[item] for item in combo], r_idx, r_id)
    groc_idx_sets_2_test.append(idx_tup)

groc_idx_sets_2_val = []
for combo, r_idx, r_id in grocery_combos_2_val:
    idx_tup = ([groc_idx[item] for item in combo], r_idx, r_id)
    groc_idx_sets_2_val.append(idx_tup)

print('finished producing grocery idx sets')

img_size = 512
min_distance = 125 # A minimum distance between image centers

print('starting mosaic creation')

for run in ['train', 'test', 'val']:
    print('starting mosaic creation for {}'.format(run))
    label_dict = {}
    label_out_fn = "mosaic_labels_e2e_2_{}.json".format(run)
    output_dir = "mosaic_e2e_2_{}".format(run)

    if run == 'train':
        groc_idx_sets_2 = groc_idx_sets_2_train
    elif run == 'test':
        groc_idx_sets_2 = groc_idx_sets_2_test
    elif run == 'val':
        groc_idx_sets_2 = groc_idx_sets_2_val

    os.makedirs(os.path.join(base_path, output_dir), exist_ok=True)
    for i, (groc_idx, r_idx, r_id) in tqdm.tqdm(enumerate(groc_idx_sets_2), desc=f'Writing {len(groc_idx_sets_2)} mosaic recipe images'):
        mosaic = manual_mosaic(groc_idx, class_samples, min_distance, img_size)
        fn = f"recipe_{r_id}.jpg"
        label_dict[fn] = {'recipe1M_id': r_id, 'recipe1M_idx': r_idx, "groc_idx": groc_idx, 'partition': run}
        cv.imwrite(os.path.join(base_path, output_dir, fn), mosaic)

    with open(os.path.join(base_path, output_dir, label_out_fn), 'w') as f:
        json.dump(label_dict, f, indent=2)
