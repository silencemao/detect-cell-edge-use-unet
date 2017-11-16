from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt
#from libtiff import TIFF

class myAugmentation(object):
    """
    A class used to augmentate image
    Firstly, read train image and label seperately, and then merge them together for the next process
    Secondly, use keras preprocessing to augmentate image
    Finally, seperate augmentated image apart into train image and label
    """

    def __init__(self, train_path="../deform/train", label_path="../deform/label", merge_path="../deform/merge",
                 aug_merge_path="../deform/aug_merge", aug_train_path="../deform/aug_train",
                 aug_label_path="../deform/aug_label", img_type="tif"):

        """
        Using glob to get all .img_type form path
        """

        self.train_imgs = glob.glob(train_path + "/*." + img_type)  # 训练集
        self.label_imgs = glob.glob(label_path + "/*." + img_type)  # label
        self.train_path = train_path
        self.label_path = label_path
        self.merge_path = merge_path
        self.img_type = img_type
        self.aug_merge_path = aug_merge_path
        self.aug_train_path = aug_train_path
        self.aug_label_path = aug_label_path
        self.slices = len(self.train_imgs)
        self.datagen = ImageDataGenerator(
            rotation_range=0.2,
            width_shift_range=0.05,
            height_shift_range=0.05,
            shear_range=0.05,
            zoom_range=0.05,
            horizontal_flip=True,
            fill_mode='nearest')

    def Augmentation(self):
        # 读入3通道的train和label, 分别转换成矩阵, 然后将label的第一个通道放在train的第2个通处, 做数据增强
        print("运行 Augmentation")
        """
        Start augmentation.....
        """
        trains = self.train_imgs
        labels = self.label_imgs
        path_train = self.train_path
        path_label = self.label_path
        path_merge = self.merge_path
        imgtype = self.img_type
        path_aug_merge = self.aug_merge_path
        print(len(trains), len(labels))
        if len(trains) != len(labels) or len(trains) == 0 or len(trains) == 0:
            print("trains can't match labels")
            return 0
        for i in range(len(trains)):
            img_t = load_img(path_train + "/" + str(i) + "." + imgtype)  # 读入train
            img_l = load_img(path_label + "/" + str(i) + "." + imgtype)  # 读入label
            x_t = img_to_array(img_t)                                    # 转换成矩阵
            x_l = img_to_array(img_l)
            x_t[:, :, 2] = x_l[:, :, 0]                                  # 把label当做train的第三个通道
            img_tmp = array_to_img(x_t)
            img_tmp.save(path_merge + "/" + str(i) + "." + imgtype)      # 保存合并后的图像
            img = x_t
            img = img.reshape((1,) + img.shape)                          # 改变shape(1, 512, 512, 3)
            savedir = path_aug_merge + "/" + str(i)                      # 存储合并增强后的图像
            if not os.path.lexists(savedir):
                os.mkdir(savedir)
            self.doAugmentate(img, savedir, str(i))                      # 数据增强

    def doAugmentate(self, img, save_to_dir, save_prefix, batch_size=1, save_format='tif', imgnum=30):
        print("运行 doAugmenttaion")
        """
        augmentate one image
        """
        datagen = self.datagen
        i = 0
        for batch in datagen.flow(img,
                                  batch_size=batch_size,
                                  save_to_dir=save_to_dir,
                                  save_prefix=save_prefix,
                                  save_format=save_format):
            i += 1
            if i > imgnum:
                break

    def splitMerge(self):
        # 读入合并增强之后的数据(aug_merge), 对其进行分离, 分别保存至 aug_train, aug_label
        print("运行 splitMerge")
        """
        split merged image apart
        """
        path_merge = self.aug_merge_path       # 合并增强之后的图像
        path_train = self.aug_train_path       # 增强之后分离出来的train
        path_label = self.aug_label_path       # 增强之后分离出来的label
        for i in range(self.slices):
            path = path_merge + "/" + str(i)
            print(path)
            train_imgs = glob.glob(path + "/*." + self.img_type)  # 所有训练图像
            savedir = path_train + "/" + str(i)                   # 保存训练集的路径
            if not os.path.lexists(savedir):
                os.mkdir(savedir)
            savedir = path_label + "/" + str(i)                   # 保存label的路径
            if not os.path.lexists(savedir):
                os.mkdir(savedir)
            for imgname in train_imgs:         # rindex("/") 是返回'/'在字符串中最后一次出现的索引
                midname = imgname[imgname.rindex("/") + 1:imgname.rindex("." + self.img_type)] # 获得文件名(不包含后缀)
                img = cv2.imread(imgname)      # 读入训练图像
                img_train = img[:, :, 2]  # 训练集是第2个通道, label是第0个通道
                img_label = img[:, :, 0]
                cv2.imwrite(path_train + "/" + str(i) + "/" + midname + "_train" + "." + self.img_type, img_train) # 保存训练图像和label
                cv2.imwrite(path_label + "/" + str(i) + "/" + midname + "_label" + "." + self.img_type, img_label)

class dataProcess(object):
    def __init__(self, out_rows, out_cols, data_path="../deform/train", label_path="../deform/label",
                 test_path="../deform/test", npy_path="../deform/npydata", img_type="tif"):

        """

        """

        self.out_rows = out_rows
        self.out_cols = out_cols
        self.data_path = data_path
        self.label_path = label_path
        self.img_type = img_type
        self.test_path = test_path
        self.npy_path = npy_path

    def create_train_data(self):
        # 将训练集和label(没有增强的数据)分别生成npy格式, 训练集是分离之后的图像
        i = 0
        print('-' * 30)
        print('Creating training images...')
        print('-' * 30)
        imgs = glob.glob(self.data_path + "/*." + self.img_type)           # deform/train
        print(len(imgs))
        imgdatas = np.ndarray((len(imgs), self.out_rows, self.out_cols, 1), dtype=np.uint8)
        imglabels = np.ndarray((len(imgs), self.out_rows, self.out_cols, 1), dtype=np.uint8)
        for imgname in imgs:
            midname = imgname[imgname.rindex("/") + 1:]   # 图像的名字
            img = load_img(self.data_path + "/" + midname, grayscale=True)   # 转换为灰度图
            label = load_img(self.label_path + "/" + midname, grayscale=True)
            img = img_to_array(img)
            label = img_to_array(label)
            imgdatas[i] = img
            imglabels[i] = label
            if i % 100 == 0:
                print('Done: {0}/{1} images'.format(i, len(imgs)))
            i += 1
        print('loading done', imgdatas.shape)
        np.save(self.npy_path + '/imgs_train.npy', imgdatas)            # 将30张训练集和30张label生成npy数据
        np.save(self.npy_path + '/imgs_mask_train.npy', imglabels)
        print('Saving to .npy files done.')

    def create_test_data(self):

        # 将训练集和label(没有增强的)分别生成npy格式, 训练集是分离之后的图像
        i = 0
        print('-' * 30)
        print('Creating training images...')
        print('-' * 30)
        imgs = glob.glob(self.test_path + "/*." + self.img_type)           # deform/train
        print(len(imgs))
        imgdatas = np.ndarray((len(imgs), self.out_rows, self.out_cols, 1), dtype=np.uint8)
        for imgname in imgs:
            midname = imgname[imgname.rindex("/") + 1:]   # 图像的名字
            img = load_img(self.test_path + "/" + midname, grayscale=True)   # 转换为灰度图
            img = img_to_array(img)
            imgdatas[i] = img
            if i % 100 == 0:
                print('Done: {0}/{1} images'.format(i, len(imgs)))
            i += 1
        print('loading done', imgdatas.shape)
        np.save(self.npy_path + '/imgs_test.npy', imgdatas)            # 将30张训练集和30张label生成npy数据
        # np.save(self.npy_path + '/imgs_mask_train.npy', imglabels)
        print('Saving to .npy files done.')

    def load_train_data(self):
        # 读入训练数据包括label_mask(npy格式), 归一化(只减去了均值)
        print('-' * 30)
        print('load train images...')
        print('-' * 30)
        imgs_train = np.load(self.npy_path + "/imgs_train.npy")
        imgs_mask_train = np.load(self.npy_path + "/imgs_mask_train.npy")
        imgs_train = imgs_train.astype('float32')
        imgs_mask_train = imgs_mask_train.astype('float32')
        imgs_train /= 255
        mean = imgs_train.mean(axis=0)
        imgs_train -= mean
        imgs_mask_train /= 255
        imgs_mask_train[imgs_mask_train > 0.5] = 1
        imgs_mask_train[imgs_mask_train <= 0.5] = 0
        return imgs_train, imgs_mask_train

    def load_test_data(self):
        print('-' * 30)
        print('load test images...')
        print('-' * 30)
        imgs_test = np.load(self.npy_path + "/imgs_test.npy")
        imgs_test = imgs_test.astype('float32')
        imgs_test /= 255
        mean = imgs_test.mean(axis=0)
        imgs_test -= mean
        return imgs_test


if __name__ == "__main__":
    aug = myAugmentation()
    aug.Augmentation()
    aug.splitMerge()
    mydata = dataProcess(512, 512)
    mydata.create_train_data()
    mydata.create_test_data()
    imgs_train, imgs_mask_train = mydata.load_train_data()
    print(imgs_train.shape, imgs_mask_train.shape)
