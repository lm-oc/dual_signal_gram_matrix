import os
import shutil

def split_images_into_folders(image_path, train_file, test_file):
    # 创建训练集和测试集的文件夹
    train_folder = os.path.join(image_path, 'train')
    test_folder = os.path.join(image_path, 'test')
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # 读取训练集文件，并将图片复制到相应的类别文件夹中
    with open(train_file, 'r') as f:
        for line in f:
            image_file, class_id = line.strip().split()
            class_folder = os.path.join(train_folder, class_id)
            os.makedirs(class_folder, exist_ok=True)
            shutil.copy(image_file, class_folder)

    # 读取测试集文件，并将图片复制到相应的类别文件夹中
    with open(test_file, 'r') as f:
        for line in f:
            image_file, class_id = line.strip().split()
            class_folder = os.path.join(test_folder, class_id)
            os.makedirs(class_folder, exist_ok=True)
            shutil.copy(image_file, class_folder)

def main():
    file_path = './'  # 假设文件保存在这个路径下
    image_path = './images'  # 假设图片保存在这个路径下

    img_info_file = os.path.join(file_path, 'flickr_style_img_info.txt')
    train_file = os.path.join(file_path, 'train.txt')
    test_file = os.path.join(file_path, 'test.txt')

    # 调用新函数，将图片划分到不同的文件夹中
    split_images_into_folders(image_path, train_file, test_file)

if __name__ == '__main__':
    main()
