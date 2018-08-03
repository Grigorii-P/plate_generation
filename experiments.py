import cv2
from os.path import join
from random import shuffle
import os
import json
import numpy as np
from time import time


path_templates = '/home/grigorii/Desktop/plates_generator/templates'
path_jsons = "/home/grigorii/Desktop/primary_search/2017-10-03T00_00_01__2017-11-01T00_00_00/"
path_to_imgs = '/home/grigorii/Desktop/primary_search/2017-10-03T00_00_01__2017-11-01T00_00_00/nn_images'
path_to_save = '/home/grigorii/Desktop/plates_generator/generated'
path_imgs_generated = '/home/grigorii/Desktop/plates_generator/generated'
path_npy = '/home/grigorii/Desktop/plates_generator/npy'

alphabet = ['A', 'B', 'C', 'E', 'H', 'K', 'M', 'O', 'P', 'T', 'X', 'Y',
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet_ru = ['А', 'В', 'С', 'Е', 'Н', 'К', 'М', 'О', 'Р', 'Т', 'Х', 'У',
              '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
area_number = (60, 126, 350, 1335) # whole area for 6 first elements on a plate
area_region = (43, 1502, 261, 1793) # region area
# sizes for the first 6 elements
lx, ly = 155, 220 # letter's width and height on a plate
dx, dy = 170, 290 # digit's width and height on a plate
dist_let_dig = 67 # distance between letter and digit
dist_dig = 37 # distance between digits
dist_let = 36 # distance between letters
# sizes for a region
dx_r, dy_r = 130, 215 # digit's width and height in a region
dist_dig_r = 29 # distance between digits
resize = (30, 150)
num_to_create = 20000


def get_plate_img(elements):
    cur_x_pos = 0
    temp = cv2.imread(join(path_templates, 'template.png'), 0)
    img = cv2.imread(join(path_templates, elements[0] + '.png'), 0)
    img = cv2.resize(img, (lx, ly))
    shift_dy = area_number[2] - ly
    shift_dx = area_number[1]
    temp[shift_dy:shift_dy + img.shape[0], shift_dx:shift_dx + img.shape[1]] = img
    cur_x_pos += shift_dx + img.shape[1] + dist_let_dig
    
    for i in range(3):
        img = cv2.imread(join(path_templates, elements[i+1] + '.png'), 0)
        img = cv2.resize(img, (dx, dy))
        #TODO put out of the loop
        shift_dy = area_number[2] - dy
        shift_dx = cur_x_pos
        temp[shift_dy:shift_dy + img.shape[0], shift_dx:shift_dx + img.shape[1]] = img
        if i != 2:
            cur_x_pos += img.shape[1] + dist_dig
        else:
            cur_x_pos += img.shape[1] + dist_let_dig
            
    for i in range(2):
        img = cv2.imread(join(path_templates, elements[i+4] + '.png'), 0)
        img = cv2.resize(img, (lx, ly))
        #TODO put out of the loop
        shift_dy = area_number[2] - ly
        shift_dx = cur_x_pos
        temp[shift_dy:shift_dy + img.shape[0], shift_dx:shift_dx + img.shape[1]] = img
        cur_x_pos += img.shape[1] + dist_let
        
    cur_x_pos = 0
    img = cv2.imread(join(path_templates, elements[6] + '.png'), 0)
    img = cv2.resize(img, (dx_r, dy_r))
    shift_dy = area_region[2] - dy_r
    shift_dx = area_region[1]
    temp[shift_dy:shift_dy + img.shape[0], shift_dx:shift_dx + img.shape[1]] = img
    cur_x_pos += shift_dx + img.shape[1] + dist_dig_r
    
    img = cv2.imread(join(path_templates, elements[7] + '.png'), 0)
    img = cv2.resize(img, (dx_r, dy_r))
    shift_dx = cur_x_pos
    temp[shift_dy:shift_dy + img.shape[0], shift_dx:shift_dx + img.shape[1]] = img
    
    return temp


def all_images_file():
    all_images = {}

    if not os.path.exists('all_images.json'):
        print('Creating all_images file')
        files = os.listdir(path_jsons)
        json_list = []
        for file in files:
            if file.endswith(".json"):
                json_list.append(file)

        shift = 5

        data_all = []
        for json_file in json_list:
            with open(join(path_jsons, json_file)) as f:
                data = json.load(f)
                data_all.append(data)
                for i, item in enumerate(data['results']):

                    # add first image from two
                    img_name = item['firstOct']['photoProof']['link'].split('/')[-1]
                    left = item['firstOct']['photoProof']['bounds']['leftBorder']
                    top = item['firstOct']['photoProof']['bounds']['topBorder']
                    right = item['firstOct']['photoProof']['bounds']['rightBorder']
                    bottom = item['firstOct']['photoProof']['bounds']['bottomBorder']
                    number = item['firstOct']['correctedCarNumber']
                    middle_part = number['middleCarNumber']
                    region_part = number['regionCarNumber'].split(' ')[0]
                    all_images[img_name] = {'coords':(left, top - shift, right, bottom + shift),
                                           'car_number':list(middle_part + region_part)}

                    # add first image from two
                    img_name = item['secondOct']['photoProof']['link'].split('/')[-1]
                    left = item['secondOct']['photoProof']['bounds']['leftBorder']
                    top = item['secondOct']['photoProof']['bounds']['topBorder']
                    right = item['secondOct']['photoProof']['bounds']['rightBorder']
                    bottom = item['secondOct']['photoProof']['bounds']['bottomBorder']
                    number = item['secondOct']['correctedCarNumber']
                    middle_part = number['middleCarNumber']
                    region_part = number['regionCarNumber'].split(' ')[0]
                    all_images[img_name] = {'coords':(left, top - shift, right, bottom + shift),
                                           'car_number':list(middle_part + region_part)}

        with open('all_images.json', 'w') as fp:
            json.dump(all_images, fp)

    else:
        print('all_images file already exists')
        with open('all_images.json', 'r') as fp:
            all_images = json.load(fp)

    return all_images


## to create test set for prediction
# path_test = '/home/grigorii/Desktop/plates_generator/test'
# path_to_save = path_test

def create_images():
    all_images = all_images_file()

    list_to_create = [key for key in all_images.keys()]
    shuffle(list_to_create)
    list_to_create = list_to_create[:num_to_create]

    num_missed_len = 0
    num_missed_NoneType = 0
    for i, item in enumerate(list_to_create):
        number = all_images[item]['car_number']
        number = [alphabet[alphabet_ru.index(x)] for x in number]
        if len(number) != 8:
            num_missed_len += 1
            continue
        img = cv2.imread(join(path_to_imgs, item), 0)
        if img is None:
            num_missed_NoneType += 1
            continue
        area = all_images[item]['coords']
        cropped_img = img[area[1]:area[3], area[0]:area[2]]
        cv2.imwrite(join(path_to_save, str(i) + '.jpg'), cropped_img)
        temp = get_plate_img(number)
        cv2.imwrite(join(path_to_save, str(i) + '_temp.jpg'), temp)

    print('Portion of len != 8: {:.2f} '.format(num_missed_len / num_to_create))
    print('Portion of NoneType: {:.5f}'.format(num_missed_NoneType / num_to_create))


def create_npy():
    if os.path.exists(join(path_npy, 'data.npy')):
        return

    images = os.listdir(path_imgs_generated)
    total = round(len(images) / 2)

    imgs_input = np.ndarray((total, resize[0] * resize[1]), dtype=np.float32)
    imgs_target = np.ndarray((total, resize[0] * resize[1]), dtype=np.float32)

    print('Creating images...')
    i = 0
    for image_trg_name in images:
        if 'temp' in image_trg_name:
            continue
        image_input_name = image_trg_name.split('.')[0] + '_temp.jpg'
        img_trg = cv2.imread(join(path_imgs_generated, image_trg_name), 0)
        img_inp = cv2.imread(join(path_imgs_generated, image_input_name), 0)
        
        
        try:
            img_trg = cv2.resize(img_trg, (resize[1], resize[0]))
            img_inp = cv2.resize(img_inp, (resize[1], resize[0]))
        except:
            print(image_trg_name)
            print(type(img_trg))
            print(image_input_name)
            print(type(img_inp))
            break
        
        img_trg = np.divide(img_trg, 255.)
        img_inp = np.divide(img_inp, 255.)
        
        img_trg = img_trg.flatten()
        img_inp = img_inp.flatten()
        img_trg = np.reshape(img_trg, (1, img_trg.shape[0]))
        img_inp = np.reshape(img_inp, (1, img_inp.shape[0]))
        
        img_trg = np.array(img_trg, dtype=np.float32)
        img_inp = np.array(img_inp, dtype=np.float32)
        
        imgs_target[i] = img_trg
        imgs_input[i] = img_inp
        i += 1

        if i % 500 == 0:
            print('Done: {0}/{1} images'.format(i * 2, total * 2))
    
    np.save(join(path_npy, 'data.npy'), imgs_target)
    np.save(join(path_npy, 'data_temp.npy'), imgs_input)


if __name__ == '__main__':
    create_images()
    create_npy()
