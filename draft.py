from data_processing import all_images_file, get_random_one_line_plate, printing
from os.path import join
import cv2


alphabet = ['A', 'B', 'C', 'E', 'H', 'K', 'M', 'O', 'P', 'T', 'X', 'Y',
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet_ru = ['А', 'В', 'С', 'Е', 'Н', 'К', 'М', 'О', 'Р', 'Т', 'Х', 'У',
              '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

path_to_save = '/ssd480/data/one_line_synthetic'

all_images = all_images_file()
num_plates = 500000
with open(join(path_to_save, 'meta.txt'), 'w') as f:
    for i in range(num_plates):
        img, number = get_random_one_line_plate()
        number = [alphabet_ru[alphabet.index(x)] for x in number]
        cv2.imwrite(join(path_to_save, str(i) + '.jpg'), img)
        f.write('{} {}\n'.format(str(i) + '.jpg', number))
