import os

data_dir = './data/'
txt_file = 'datalist.txt'


def get_data_path():

    class_num = 0
    img_label_list = []

    categorys = os.listdir(data_dir)
    for category in categorys:
        # Get all files in current category's folder.
        folder_path = os.path.join(data_dir, category)  # e.g. './data/cat/'
        imagenames = sorted(os.listdir(folder_path))

        for image in imagenames:
            imagepath = os.path.join(folder_path, image)
            _img_label = imagepath + ' ' + str(class_num)
            img_label_list.append(_img_label)
        class_num += 1

    write_txt(txt_file, img_label_list)


def write_txt(file, image):

    txt = open(file, 'w')
    num = 0
    for i in image:
        t = i + '\n'
        txt.writelines(t)
        num = num + 1
    txt.close()


if __name__ == "__main__":
    print('Making dataset txt file.')
    get_data_path()