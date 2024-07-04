import os  #用于对文件夹的系列操作
from os.path import join
import torch
import torch.utils.data as data
from torchvision.transforms import transforms
from PIL import Image

#定义训练、测试及验证数据的图片预处理
# train_transform = transforms.Compose([   #transform的系列操作，建议参考https://zhuanlan.zhihu.com/p/53367135
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomResizedCrop(32),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])
# train_transform = transforms.Compose([
#     transforms.RandomResizedCrop(224),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])
# test_transform = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.Resize(45),
#     transforms.CenterCrop(32),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_transform = transforms.Compose([
    transforms.Resize(45),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
#定义后缀，用来对图片的查找与读取
#曾出现过错误ValueError: num_samples should be a positive integer value, but got num_samples=0
#因为后缀出错，读取不到图片
#此处需要用（），不能用[]，不能是列表数据
FileNameEnd = ('.jpeg', '.JPEG', '.tif', '.jpg', '.png', '.bmp')

class ImageFolder(data.Dataset):
    def __init__(self, root, subdir='train', transform=None):
        super(ImageFolder,self).__init__()
        self.transform = transform  #定义图片转换的类型
        self.image = []     #定义存储图片的列表
        #首先需要得到训练图片的最终路径，用来读取图片，同时需要得到图片对应的文件夹的名称，最为标签数据
        #因此在制作数据集之前，图片存放路径及各个文件夹的命名需要规范
        train_dir = join(root, 'train')  #注意此处不能使用subdir，因为之后的某些值在test及val中也需要使用
        #获取训练文件夹的路径后，train文件夹下面为各种标签命名的文件夹，读取名称作为标签数据
        #sorted可以用来根据名称对读取后的数据排序，得到列表数据
        self.class_names = sorted(os.listdir(train_dir))
        #然后将class_names排序，变成字典，并将序号值与文件夹名称调换位置，使得文件夹名称变为字典的keys数据，数字类型的序号变为values数据
        self.names2index = {v: k for k, v in enumerate(self.class_names)}
        #以上算是制作标签数据的完成，之后需要根据训练、验证、测试数据来具体分析
        #大致的思路是，获取图片具体路径，并将其与标签一一对应，得到多个数组，存入self.image中制作成列表
        #比如self.image[1]可以检索到第二张图片的路径，以及第二张图片的标签形成的数组
        if subdir == 'train':
            for label in self.class_names:
                # 获取文件夹路径，我的路径为：D:/Anaconda3/data/tiny-imagenet-200/train/n01443537
                #其中n01443537为图片对应的文件夹名称，即为标签
                d = join(root, subdir, label)
                #os.walk的用法，遍历文件夹，获取文件的路径，子文件夹的名称，以及文件的名称
                #其中directory为文件夹的初始路径，_表示子文件夹名称，names则是文件名称
                #需要根据具体情况进行修改
                for directory, _, names in os.walk(d):
                    for name in names:
                        filename = join(directory, name)
                        if filename.endswith(FileNameEnd):
                            # 注意此处的双括号，append()可以把数据加到列表后，此处需要的是把数组加进去，因此有append(())
                            self.image.append((filename, self.names2index[label]))

#验证数据
        #验证数据中的标签数据并不是文件夹名称，存放在txt文档中，因此需要读取txt文档
        if subdir == 'val':
            val_dir = join(root, subdir)
            with open(join(val_dir, 'val_annotations.txt'), 'r') as f:
                infos = f.read().strip().split('\n')[:10000]
                infos = [info.strip().split('\t')[:2] for info in infos]
                self.image = [(join(val_dir, 'images', info[0]), self.names2index[info[1]]) for info in infos]
        #测试数据的读取，测试数据仍然读取的是val文件夹下面的图片，因此test文件下的图片没有被使用
        if subdir == 'test':
            test_dir = join(root, 'val')
            with open(join(test_dir, 'val_annotations.txt'), 'r') as f:
                infos = f.read().strip().split('\n')[10000:]
                infos = [info.strip().split('\t')[:2] for info in infos]
                self.image = [(join(test_dir, 'images', info[0]), self.names2index[info[1]]) for info in infos]

    def __getitem__(self, item):
        path, label = self.image[item]
        with open(path, 'rb') as f:  # rb读取二进制文件
            img = Image.open(f).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.image)

#测试下验证数据集中的标签能否与训练数据集中的标签对应
if __name__ == '__main__':
    TestData = ImageFolder('./data/tiny-imagenet-200', subdir='train', transform=train_transform)
    with open('./data/tiny-imagenet-200/val/val_annotations.txt', 'r') as f:
        infos = f.read().strip().split('\n')
        infos = [info.strip().split('\t')[1] for info in infos]
        for classname in infos:
            if not (classname in TestData.names2index):
                print('Sorry!!!')
        print('Yes!!!')