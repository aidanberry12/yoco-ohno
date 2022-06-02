
grocery_to_ingredient = 
{'Apple': ['apple', 'applejack', 'pineapple', 'crabapples', 'applesauce', 'fiber_supplement'],
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

 ingredient_to_grocery = {v:k for k, v in grocery_to_ingredient.items()}


# map a list of grocery ingredients to corresponding recipe1M ingredients 
def map_image_ingredient(ingr_list):
  
    mapper = grocery_to_ingredient

    mapped_ingr = []
    for food_item in ingr_list:
        if food_item in mapper.keys():
            lookup = mapper[food_item]
            mapped_ingr.extend(lookup)
            
    return mapped_ingr