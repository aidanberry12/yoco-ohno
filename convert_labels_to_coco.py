# %%
import json
import os
import shutil
from tqdm import tqdm
# %%
base_path = "./ML_Decoder/mosaic"
with open(os.path.join(base_path, 'mosaic_labels.json')) as f:
    label_dict = json.load(f)

# %%
# Iterate over our keys and convert them to MSCOCO-style annotations
class_names = [
    "Apple",
    "Avocado",
    "Banana",
    "Kiwi",
    "Lemon",
    "Lime",
    "Mango",
    "Melon",
    "Nectarine",
    "Orange",
    "Papaya",
    "Passion-Fruit",
    "Peach",
    "Pear",
    "Pineapple",
    "Plum",
    "Pomegranate",
    "Red-Grapefruit",
    "Satsumas",
    "Juice",
    "Milk",
    "Oatghurt",
    "Oat-Milk",
    "Sour-Cream",
    "Sour-Milk",
    "Soyghurt",
    "Soy-Milk",
    "Yoghurt",
    "Asparagus",
    "Aubergine",
    "Cabbage",
    "Carrots",
    "Cucumber",
    "Garlic",
    "Ginger",
    "Leek",
    "Mushroom",
    "Onion",
    "Pepper",
    "Potato",
    "Red-Beet",
    "Tomato",
    "Zucchini"
]

label_mapping = [{"id": i, "name": entry} for i, entry in enumerate(class_names)]

train_coco_dict = {
    "info": {
        "year": "2022",
        "version": "1.0",
        "description": "Grocery store mosaic training dataset for multi-label classification",
        "contributor": "Lyle Scott Brown",
        "date_created": "2022-04-09T00:53:50"
    },
    "categories": label_mapping,
    "images": [],
    "annotations": []
}

val_coco_dict = {
    "info": {
        "year": "2022",
        "version": "1.0",
        "description": "Grocery store mosaic validation dataset for multi-label classification",
        "contributor": "Lyle Scott Brown",
        "date_created": "2022-04-09T00:53:50"
    },
    "categories": label_mapping,
    "images": [],
    "annotations": []
}

images = []
img_id = 0
label_id = 0
os.makedirs(os.path.join(base_path, "train"), exist_ok=True)
os.makedirs(os.path.join(base_path, "val"), exist_ok=True)

for k, v in tqdm(label_dict.items(), desc="Writing COCO annotation and moving images"):
    img_entry = {
        "id": img_id,
        "file_name": k,
        "height": 512,
        "width": 512
    }

    for i in range(len(v['labels'])):
        label_entry = {
            "id": label_id,
            "image_id": img_id,
            "category_id":  v['labels'][i],
            "area": v['areas'][i]
        }
        label_id += 1
        
        if v["split"] == "train":
            train_coco_dict['annotations'].append(label_entry)
        else:
            val_coco_dict['annotations'].append(label_entry)
    
    if v["split"] == "train":
        train_coco_dict['images'].append(img_entry)
        shutil.copyfile(os.path.join(base_path, k), os.path.join(base_path, "train", k))
    else:
        val_coco_dict['images'].append(img_entry)
        shutil.copyfile(os.path.join(base_path, k), os.path.join(base_path, "val", k))

    img_id += 1

with open(os.path.join(base_path, "instances_train.json"), 'w') as f:
    json.dump(train_coco_dict, f, indent=2)

with open(os.path.join(base_path, "instances_val.json"), 'w') as f:
    json.dump(val_coco_dict, f, indent=2)


# %%
