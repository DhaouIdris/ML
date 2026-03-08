import json 
import numpy as np 
import cv2
import matplotlib.pyplot as plt


def get_dimensions(data):
    file_data = list(data.values())[0]
    regions = file_data['regions']
    
    max_x = 0
    max_y = 0

    for region in regions:
        shape = region['shape_attributes']
        if shape["name"]== "polygon":
            max_x = max(max(shape['all_points_x']), max_x)
            max_y = max(max(shape['all_points_y']), max_y)
    return max_x + 1, max_y + 1

# On crée un mask avec les dimensions de l'image, et on attribue à chaque pixel une classe
# 0 pour la membrane, 1 pour la zone incertaine, et 2 pour les bactéries

def create_mask(data):
    file_data = list(data.values())[0] 
    regions = file_data['regions']
    
    height, width = get_dimensions(data)

    mask = np.zeros((width, height), dtype=np.uint8)

    class_mapping = {   
    "unsure": 0,
    "membrane": 125,
    "bacteria": 255 }

    order = ['membrane','unsure',"bacteria"]

    # Sort regions
    regions.sort(key=lambda r: order.index(r['region_attributes']['class']) 
                 if r['region_attributes']['class'] in order else -1)
    for region in regions:

        class_names = region['region_attributes']['class']

        class_id = class_mapping.get(class_names, 0)

        shape = region['shape_attributes']

        X = shape['all_points_x']
        Y = shape['all_points_y']

        points = np.array(list(zip(X, Y)), dtype=np.int32)
        points = points.reshape((-1, 1, 2))


        cv2.fillPoly(mask, [points], class_id)

    return mask





if __name__ == "__main__":
    #load data from json file
    with open('./annotations.json') as f:
        data = json.load(f)
    
    mask = create_mask(data)

    mask = mask
    cv2.imwrite("mask.png", mask)
    print("\nSaved 'mask.png' - open this to see the classes visually.")

    plt.imshow(mask, cmap='gray')
    plt.title('Mask Image')
    plt.show()