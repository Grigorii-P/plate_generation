import cv2
from os.path import join, exists
from random import shuffle, choice, randint, uniform
import os
import json
import numpy as np
from utils.unet import img_cols, img_rows
from time import time


# path_templates = '/home/grigorii/Desktop/plates_generator/templates'
# path_jsons = "/home/grigorii/Desktop/2017-10-03T00_00_01__2017-11-01T00_00_00/"
# path_nn_imgs = '/home/grigorii/Desktop/2017-10-03T00_00_01__2017-11-01T00_00_00/nn_images'
# path_to_save = '/home/grigorii/Desktop/plates_generator/generated'
# path_imgs_generated = '/home/grigorii/Desktop/plates_generator/generated'
# path_npy = '/home/grigorii/Desktop/plates_generator/npy'
# path_to_cascade = "/home/grigorii/ssd480/talos/python/platedetection/haar/cascade_inversed_plates.xml"
path_templates = '/ssd480/grisha/plates_generation/templates'
path_jsons = "/ssd480/data/metadata/"
path_nn_imgs = '/ssd480/data/nn_images'
path_to_save = '/ssd480/grisha/plates_generation/generated_400000_cropped_VJ'
path_imgs_generated = '/ssd480/grisha/plates_generation/generated_400000_cropped_VJ'
path_npy = '/ssd480/grisha/plates_generation/npy'
path_to_cascade = "/ssd480/talos/python/platedetection/haar/cascade_inversed_plates.xml"

alphabet = ['A', 'B', 'C', 'E', 'H', 'K', 'M', 'O', 'P', 'T', 'X', 'Y',
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet_ru = ['А', 'В', 'С', 'Е', 'Н', 'К', 'М', 'О', 'Р', 'Т', 'Х', 'У',
              '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet_letters = ['A', 'B', 'C', 'E', 'H', 'K', 'M', 'O', 'P', 'T', 'X', 'Y']
digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

shift = 5 # increase borders when cropping plates

# TODO group parameters on `usual number` and `two-line number`
# usual vehicle number
area_number = (60, 126, 350, 1335) # whole area for 6 first elements on a plate
area_region_2 = (43, 1502, 261, 1793) # region area
area_region_3 = (43, 1376, 261, 1818) # region area
# two line vehicle number
area_number_two_line = (39, 59, 131, 328) # whole area for 6 first elements on a plate
area_two_line_region_2 = (179, 24, 273, 177) # region area
area_two_line_region_3 = (179, 218, 273, 346) # region area
# sizes for the first 6 elements
lx, ly = 155, 220 # letter's width and height on a plate
dx, dy = 170, 290 # digit's width and height on a plate

# digit's width and height
dx_d_two_line = 57
dy_d_two_line = area_number_two_line[2] - area_number_two_line[0]
dx_l_two_line = 73 - 5
dy_l_two_line = area_two_line_region_2[2] - area_two_line_region_2[0] - 5
dx_r_two_line = 62 - 2
dy_r_two_line = dy_l_two_line

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
minSize_ = (50, 10)
maxSize_ = (200, 40)

resize = (30, 150)
num_to_create_in_generated_folder = 500000
num_train = 10000
num_valid = 100
num_imgs_unet = 200000


def printing(s):
    print('-' * 30)
    print(s)
    print('-' * 30)


def get_one_line_plate_img(elements, is_region_3):
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
        # TODO put out of the loop
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
        # TODO put out of the loop
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


def get_random_one_line_plate():
    p_3_reg = 0.3
    num = []
    num.append(choice(alphabet_letters))
    for j in range(3):
        num.append(choice(digits))
    for j in range(2):
        num.append(choice(alphabet_letters))
    if uniform(0, 1) < p_3_reg:
        reg = list(str(randint(100, 999)))
        flag = True
    else:
        reg = list(str(randint(1, 100)))
        flag = False
    num.extend(reg)
    img = get_one_line_plate_img(num, flag)
    return img, num


def get_two_line_plate_img(elements):
    template_name = 'template_two_line.jpg'
    temp = cv2.imread(join(path_templates, template_name), 0)

    n = 4
    dist = round((area_number_two_line[3] - area_number_two_line[1] - n * dx_d_two_line) / (n - 1))
    shift_dy = area_number_two_line[2] - dy_d_two_line
    shift_dx = area_number_two_line[1]
    cur_x_pos = shift_dx
    for i in range(n):
        img = cv2.imread(join(path_templates, elements[i] + '.png'), 0)
        img = cv2.resize(img, (dx_d_two_line, dy_d_two_line))
        temp[shift_dy:shift_dy + img.shape[0], cur_x_pos:cur_x_pos + img.shape[1]] = img
        cur_x_pos += img.shape[1] + dist

    n = 2
    dist = round((area_two_line_region_2[3] - area_two_line_region_2[1] - n * dx_l_two_line) / (n - 1))
    shift_dy = area_two_line_region_2[2] - dy_l_two_line
    shift_dx = area_two_line_region_2[1]
    cur_x_pos = shift_dx
    for i in range(n):
        img = cv2.imread(join(path_templates, elements[4 + i] + '.png'), 0)
        img = cv2.resize(img, (dx_l_two_line, dy_l_two_line))
        temp[shift_dy:shift_dy + img.shape[0], cur_x_pos:cur_x_pos + img.shape[1]] = img
        cur_x_pos += img.shape[1] + dist

    n = 2
    dist = round((area_two_line_region_3[3] - area_two_line_region_3[1] - n * dx_r_two_line) / (n - 1))
    shift_dy = area_two_line_region_3[2] - dy_r_two_line
    shift_dx = area_two_line_region_3[1]
    cur_x_pos = shift_dx
    for i in range(n):
        img = cv2.imread(join(path_templates, elements[6 + i] + '.png'), 0)
        img = cv2.resize(img, (dx_r_two_line, dy_r_two_line))
        temp[shift_dy:shift_dy + img.shape[0], cur_x_pos:cur_x_pos + img.shape[1]] = img
        cur_x_pos += img.shape[1] + dist

    return temp


def all_images_file():
    all_images = {}

    if not os.path.exists('all_images.json'):
        printing('Creating all_images file...')
        files = os.listdir(path_jsons)
        json_list = []
        for file in files:
            if file.endswith(".json"):
                json_list.append(file)

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
        printing('`all_images` file already exists')
        with open('all_images.json', 'r') as fp:
            all_images = json.load(fp)

    return all_images


def check_img_dimensions(img):
    for item in img.shape:
        if item == 0:
            raise ValueError('img dimension is wrong - {}'.format(img.shape))


def create_images():
    check = os.listdir(path_imgs_generated)
    if check:
        printing('images in `generated` folder already exist')
        return

    all_images = all_images_file()
    plate_cascade = cv2.CascadeClassifier(path_to_cascade)
    printing('Creating images in `{}` folder...'.format(path_imgs_generated))

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
        check_area(area)
        cropped_img = img[area[1]:area[3], area[0]:area[2]]
        check_img_dimensions(cropped_img)
        
        # in case num elem on plate < 7 or > 9,
        # `num_steps` inside `get_one_line_plate_img` dict
        # throws an exception
        try:
            temp = get_one_line_plate_img(number, flag)
        except:
            continue

        plates = plate_cascade.detectMultiScale(cropped_img, scaleFactor=1.3, minNeighbors=3,
                                                minSize=minSize_,
                                                maxSize=maxSize_)
        # TODO add to `generator`
        if len(plates) != 0:
            (x, y, w, h) = choice(plates)
            cropped_img = cropped_img[y:y + h, x:x + w]

        cv2.imwrite(join(path_to_save, item), cropped_img)
        cv2.imwrite(join(path_to_save, item.split('.')[0] + '_temp.jpg'), temp)

        if i % 10000 == 0:
            print('{} plates created out of {}'.format(i, num_to_create_in_generated_folder))

    printing('Portion of NoneType: {:.3f}'.format(num_missed_NoneType / num_to_create_in_generated_folder))


def create_npy_autoenc():
    if os.path.exists(join(path_npy, 'validation_trg.npy')):
        printing('npy files already exist')
        return

    images = os.listdir(path_imgs_generated)
    total = round(len(images) / 2)

    printing('Creating npy files...')
    i = 0
    list_imgs_trg = []
    list_imgs_inp = []
    for image_trg_name in images:
        if 'temp' in image_trg_name:
            continue
        image_input_name = image_trg_name.split('.')[0] + '_temp.jpg'
        img_trg = cv2.imread(join(path_imgs_generated, image_trg_name), 0)
        img_inp = cv2.imread(join(path_imgs_generated, image_input_name), 0)

        # TODO you may delete try except (because we already have check_dimensions)
        try:
            img_trg = cv2.resize(img_trg, (resize[1], resize[0]))
            img_inp = cv2.resize(img_inp, (resize[1], resize[0]))
        except:
            print('Resize error')
            print('img_trg shape - {}'.format(img_trg.shape))
        
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

        if i % 1000 == 0:
            print('Done: {0}/{1} images'.format(i * 2, total * 2))

    imgs_input = np.ndarray((total, resize[0] * resize[1]), dtype=np.float32)
    imgs_target = np.ndarray((total, resize[0] * resize[1]), dtype=np.float32)

    for i, item in enumerate(list_imgs_trg):
        imgs_target[i] = item
    for i, item in enumerate(list_imgs_inp):
        imgs_input[i] = item

    np.save(join(path_npy, 'validation_trg.npy'), imgs_target)
    np.save(join(path_npy, 'validation_inp.npy'), imgs_input)


def create_npy_unet():
    if os.path.exists(join(path_npy, 'input_unet.npy')):
        printing('npy files already exist')
        return

    images = os.listdir(path_imgs_generated)
    images.sort()
    images = images[:num_imgs_unet]
    total = round(len(images) / 2)

    printing('Creating npy files...')
    i = 0
    list_imgs_trg = []
    list_imgs_inp = []

    for c, image_trg_name in enumerate(images):
        if 'temp' in image_trg_name:
            continue
        image_input_name = image_trg_name.split('.')[0] + '_temp.jpg'
        img_trg = cv2.imread(join(path_imgs_generated, image_trg_name), 0)
        img_inp = cv2.imread(join(path_imgs_generated, image_input_name), 0)

        # TODO you may delete try except (because we already have check_dimensions)
        try:
            img_trg = cv2.resize(img_trg, (img_rows, img_cols))
            img_inp = cv2.resize(img_inp, (img_rows, img_cols))
        except:
            print('Resize error')
            print('img_trg shape - {}'.format(img_trg.shape))

        img_trg = np.divide(img_trg, 255.)
        img_inp = np.divide(img_inp, 255.)

        img_trg = np.array(img_trg, dtype=np.float32)
        img_inp = np.array(img_inp, dtype=np.float32)

        list_imgs_trg.append(img_trg)
        list_imgs_inp.append(img_inp)

        i += 1

        if i % 1000 == 0:
            print('Done: {0}/{1} images'.format(i * 2, total * 2))

    imgs_input = np.ndarray((len(list_imgs_inp), img_rows, img_cols), dtype=np.float32)
    imgs_target = np.ndarray((len(list_imgs_trg), img_rows, img_cols), dtype=np.float32)

    for i, item in enumerate(list_imgs_trg):
        imgs_target[i] = item
    for i, item in enumerate(list_imgs_inp):
        imgs_input[i] = item

    imgs_input = imgs_input[..., np.newaxis]
    imgs_target = imgs_target[..., np.newaxis]

    print('imgs_target shape - {}, imgs_input shape - {}'.format(imgs_target.shape, imgs_input.shape))

    np.save(join(path_npy, 'input_unet.npy'), imgs_input)
    np.save(join(path_npy, 'target_unet.npy'), imgs_target)


def load_npy():
    imgs_input = np.load(join(path_npy, 'validation_inp.npy'))
    imgs_target = np.load(join(path_npy, 'validation_trg.npy'))
    return imgs_input, imgs_target


def check_images_existence(images):
    count = 0
    len_before = len(images)
    new_list = images.copy()
    for item in new_list:
        if not exists(join(path_nn_imgs, item)):
            images.remove(item)
            count += 1
    len_after = len(images)
    printing('images before - {}, images after - {}, difference - {}'.format(len_before, len_after, count))


def check_area(area_):
    # some areas extracted from json have negative values (what the fuck)
    if area_[0] < 0:
        area_[0] = 0
    if area_[1] < 0:
        area_[1] = 0


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

                img_trg = cv2.imread(join(path_nn_imgs, item), 0)
                area = images_dict[item]['coords']
                check_area(area)
                img_trg = img_trg[area[1]:area[3], area[0]:area[2]]
                check_img_dimensions(img_trg)
                
                if len(number) == 6:
                    number.append(choice(digits))
                if len(number) < 6 or len(number) > 9:
                    raise ValueError('num elem on plate is {}'.format(len(number)))
                img_inp = get_one_line_plate_img(number, flag)

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
