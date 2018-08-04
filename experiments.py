import cv2
from os.path import join, exists
from random import shuffle
import os
import json
import numpy as np
from time import time


# path_templates = '/home/grigorii/Desktop/plates_generator/templates'
# path_jsons = "/home/grigorii/Desktop/primary_search/2017-10-03T00_00_01__2017-11-01T00_00_00/"
# path_nn_imgs = '/home/grigorii/Desktop/primary_search/2017-10-03T00_00_01__2017-11-01T00_00_00/nn_images'
# path_to_save = '/home/grigorii/Desktop/plates_generator/generated'
# path_imgs_generated = '/home/grigorii/Desktop/plates_generator/generated'
# path_npy = '/home/grigorii/Desktop/plates_generator/npy'
path_templates = '/ssd480/grisha/plates_generation/templates'
path_jsons = "/ssd480/data/metadata/"
path_nn_imgs = '/ssd480/data/nn_images'
path_to_save = '/ssd480/grisha/plates_generation/generated'
path_imgs_generated = '/ssd480/grisha/plates_generation/generated'
path_npy = '/ssd480/grisha/plates_generation/npy'

alphabet = ['A', 'B', 'C', 'E', 'H', 'K', 'M', 'O', 'P', 'T', 'X', 'Y',
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet_ru = ['А', 'В', 'С', 'Е', 'Н', 'К', 'М', 'О', 'Р', 'Т', 'Х', 'У',
              '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
area_number = (60, 126, 350, 1335) # whole area for 6 first elements on a plate
area_region_2 = (43, 1502, 261, 1793) # region area
area_region_3 = (43, 1376, 261, 1818) # region area
# sizes for the first 6 elements
lx, ly = 155, 220 # letter's width and height on a plate
dx, dy = 170, 290 # digit's width and height on a plate
# when digits num of region <= 2
dist_let_dig_2 = 67 # distance between letter and digit
dist_dig_2 = 37 # distance between digits
dist_let_2 = 36 # distance between letters
# when digits num of region == 3
dist_let_dig_3 = 50
dist_dig_3 = 30
dist_let_3 = 30
# sizes for a region
dx_r, dy_r = 130, 215 # digit's width and height in a region
dist_dig_r_2 = 29 # distance between digits (num region digits <= 2)
dist_dig_r_3 = 29 # distance between digits (-//- == 3)

num_steps = {7: 1, 8: 2, 9: 3}
resize = (30, 150)
num_to_create_in_generated_folder = 100
num_train = 200000
num_valid = 100


def assertion(elements, flag):
    if len(elements) == 9 and flag:
        return True
    if (len(elements) == 7 or len(elements) == 8) and not flag:
        return True
    return False


def get_plate_img(elements, is_region_3):
    assert assertion(elements, is_region_3)

    if is_region_3:
        dist_let_dig = dist_let_dig_3
        dist_dig = dist_dig_3
        dist_let = dist_let_3
        area_region = area_region_3
        template_name = 'template_3_reg.png'
    else:
        dist_let_dig = dist_let_dig_2
        dist_dig = dist_dig_2
        dist_let = dist_let_2
        area_region = area_region_2
        template_name = 'template_2_reg.png'

    cur_x_pos = 0
    temp = cv2.imread(join(path_templates, template_name), 0)
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

    len_num = len(elements)
    n = num_steps[len_num]
    dist = round((area_region[3] - area_region[1] - n * dx_r) / (n + 1))
    shift_dy = area_region[2] - dy_r
    shift_dx = area_region[1]
    cur_x_pos = shift_dx + dist
    if n == 2:
        dist = 30  # increase distance in case of two numbers
    for i in range(n):
        img = cv2.imread(join(path_templates, elements[6 + i] + '.png'), 0)
        img = cv2.resize(img, (dx_r, dy_r))
        temp[shift_dy:shift_dy + img.shape[0], cur_x_pos:cur_x_pos + img.shape[1]] = img
        cur_x_pos += img.shape[1] + dist
    
    return temp


def all_images_file():
    all_images = {}

    if not os.path.exists('all_images.json'):
        print('Creating all_images file...')
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


def create_images():
    check = os.listdir(path_imgs_generated)
    if check:
        print('images in `generated` folder already exist')
        return

    all_images = all_images_file()
    print('Creating images in `generated` folder...')

    list_to_create = [key for key in all_images.keys()]
    shuffle(list_to_create)
    list_to_create = list_to_create[:num_to_create_in_generated_folder]

    num_missed_NoneType = 0
    for i, item in enumerate(list_to_create):
        number = all_images[item]['car_number']
        number = [alphabet[alphabet_ru.index(x)] for x in number]

        if len(number) > 8:
            flag = True
        else:
            flag = False

        img = cv2.imread(join(path_nn_imgs, item), 0)

        if img is None:
            num_missed_NoneType += 1
            continue

        area = all_images[item]['coords']
        cropped_img = img[area[1]:area[3], area[0]:area[2]]
        cv2.imwrite(join(path_to_save, str(i) + '.jpg'), cropped_img)
        temp = get_plate_img(number, flag)
        cv2.imwrite(join(path_to_save, str(i) + '_temp.jpg'), temp)

    print('Portion of NoneType: {:.5f}'.format(num_missed_NoneType / num_to_create_in_generated_folder))


def create_npy():
    if os.path.exists(join(path_npy, 'data.npy')):
        print('npy files already exist')
        return

    images = os.listdir(path_imgs_generated)
    total = round(len(images) / 2)

    print('Creating npy files...')
    i = 0
    list_imgs_trg = []
    list_imgs_inp = []
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
            continue
        
        img_trg = np.divide(img_trg, 255.)
        img_inp = np.divide(img_inp, 255.)
        
        img_trg = img_trg.flatten()
        img_inp = img_inp.flatten()
        img_trg = np.reshape(img_trg, (1, img_trg.shape[0]))
        img_inp = np.reshape(img_inp, (1, img_inp.shape[0]))
        
        img_trg = np.array(img_trg, dtype=np.float32)
        img_inp = np.array(img_inp, dtype=np.float32)

        list_imgs_trg.append(img_trg)
        list_imgs_inp.append(img_inp)

        i += 1

        if i % 500 == 0:
            print('Done: {0}/{1} images'.format(i * 2, total * 2))

    imgs_input = np.ndarray((total, resize[0] * resize[1]), dtype=np.float32)
    imgs_target = np.ndarray((total, resize[0] * resize[1]), dtype=np.float32)

    for i, item in enumerate(list_imgs_trg):
        imgs_target[i] = item
    for i, item in enumerate(list_imgs_inp):
        imgs_input[i] = item

    np.save(join(path_npy, 'validation_trg.npy'), imgs_target)
    np.save(join(path_npy, 'validation_inp.npy'), imgs_input)


def load_npy():
    imgs_target = np.load(join(path_npy, 'validation_trg.npy'))
    imgs_input = np.load(join(path_npy, 'validation_inp.npy'))
    return imgs_input, imgs_target


def check_images_existence(images):
    count = 0
    new_list = images.copy()
    for item in new_list:
        if not exists(join(path_nn_imgs, item)):
            images.remove(item)
            count += 1
    print('{} image(-s) deleted from dataset'.format(count))


def import_images_train_valid():
    all_images = all_images_file()
    images = list(all_images.keys())
    shuffle(images)
    images_train = images[:num_train]
    shuffle(images)
    images_valid = images[:num_valid]
    check_images_existence(images_train)
    check_images_existence(images_valid)
    images_dict = {x: all_images[x] for x in images_train}
    return images_train, images_valid, images_dict


def generator(batch_size, images_train, images_dict):
    while True:
        shuffle(images_train)
        for i in range(0, len(images_train), batch_size):
            batch_list = images_train[i:i + batch_size]
            imgs_input = np.ndarray((batch_size, resize[0] * resize[1]), dtype=np.float32)
            imgs_target = np.ndarray((batch_size, resize[0] * resize[1]), dtype=np.float32)

            for j, item in enumerate(batch_list):
                number = images_dict[item]['car_number']
                number = [alphabet[alphabet_ru.index(x)] for x in number]

                if len(number) > 8:
                    flag = True
                else:
                    flag = False

                # if img is None:
                #     num_missed_NoneType += 1
                #     continue

                img_trg = cv2.imread(join(path_nn_imgs, item), 0)
                img_inp = get_plate_img(number, flag)

                img_trg = cv2.resize(img_trg, (resize[1], resize[0]))
                img_inp = cv2.resize(img_inp, (resize[1], resize[0]))

                img_trg = np.divide(img_trg, 255.)
                img_inp = np.divide(img_inp, 255.)

                img_trg = img_trg.flatten()
                img_inp = img_inp.flatten()

                img_trg = np.reshape(img_trg, (1, img_trg.shape[0]))
                img_inp = np.reshape(img_inp, (1, img_inp.shape[0]))

                img_trg = np.array(img_trg, dtype=np.float32)
                img_inp = np.array(img_inp, dtype=np.float32)

                imgs_target[j] = img_trg
                imgs_input[j] = img_inp

            yield (imgs_input, imgs_target)


# if __name__ == '__main__':
#     create_images()
#     create_npy()
