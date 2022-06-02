# YOCO-OHNO: Recipe Generation from Raw Food Ingredient Images

Image to recipe networks are well-explored in deep learning. Most prior research in this space has focused on converting images of completed dishes into recipes that
match the images. However, many home cooks find themselves in a situation where they do not know what they want to cook and they have limited ingredients on hand. Therefore, our project is to build a series of deep learning models that generates a recipe from an image of raw ingredients. We used an end-to-end deep learning approach consisting of multiple modules stitched together including multi-label image classification, natural language processing , and conditional language models. The results produced were not ground breaking due to the limited labeled data that is available for food ingredients, but we propose an innovative end-to-end training architecture for this ingredient image to recipe generation process using tiled mosaics of images for training.

This proposed method solves the problem of a home chef having a limited set of ingredients and not knowing what they can make with them. Our approach was to allow a home chef to take a picture of all the ingredients they have (one image) and generate a new recipe that contains these ingredients. The task of generating recipes from raw ingredient images requires the use of both a computer vision (CV) model to identify the ingredients in the image and natural language processing (NLP) to produce the output recipe given the identified ingredients in the image.


