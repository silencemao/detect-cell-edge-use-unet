# @silence 20170516
# 这个文件是用于将增强后的train和label分别合并成npy文件, 模仿原作者的data.py
import os, glob
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

class dataProcess(object):
    def __init__(self, out_rows, out_cols, aug_merge_path="../deform/aug_merge", aug_train_path="../deform/aug_train",
                 aug_label_path="../deform/aug_label", npy_path="../deform/npydata", img_type="tif"):
        self.out_rows = out_rows
        self.out_cols = out_cols
        self.aug_merge_path = aug_merge_path
        self.aug_train_path = aug_train_path
        self.aug_label_path = aug_label_path
        self.npy_path = npy_path
        self.img_type = img_type

    def create_train_data(self):
        i = 0
        print('-' * 30)
        print('creating train image')
        print('-' * 30)
        count = 0
        for indir in os.listdir(self.aug_merge_path):
            path = os.path.join(self.aug_merge_path, indir)
            count += len(os.listdir(path))
        imgdatas = np.ndarray((count, self.out_rows, self.out_cols, 1), dtype=np.uint8)
        imglabels = np.ndarray((count, self.out_rows, self.out_cols, 1), dtype=np.uint8)
        for indir in os.listdir(self.aug_merge_path):
            trainPath = os.path.join(self.aug_train_path, indir)
            labelPath = os.path.join(self.aug_label_path, indir)
            print(trainPath, labelPath)
            imgs = glob.glob(trainPath + '/*' + '.tif')
            for imgname in imgs:
                trainmidname = imgname[imgname.rindex('/') + 1:]
                labelimgname = imgname[imgname.rindex('/') + 1:imgname.rindex('_')] + '_label.tif'
                print(trainmidname, labelimgname)
                img = load_img(trainPath + '/' + trainmidname, grayscale=True)
                label = load_img(labelPath + '/' + labelimgname, grayscale=True)
                img = img_to_array(img)
                label = img_to_array(label)
                imgdatas[i] = img
                imglabels[i] = label
                if i % 100 == 0:
                    print('Done: {0}/{1} images'.format(i, len(imgs)))
                i += 1
                print(i)
        print('loading done', imgdatas.shape)
        np.save(self.npy_path + '/augimgs_train.npy', imgdatas)            # 将30张训练集和30张label生成npy数据
        np.save(self.npy_path + '/augimgs_mask_train.npy', imglabels)
        print('Saving to .npy files done.')

    def load_train_data(self):
        print('-' * 30)
        print('loading train data')
        print('-' * 30)
        augimgs_train = np.load(self.npy_path + '/augimgs_train.npy')
        augimgs_mask_train = np.load(self.npy_path + '/augimgs_mask_train.npy')
        augimgs_train = augimgs_train.astype('float32')
        augimgs_mask_train = augimgs_mask_train.astype('float32')
        augimgs_train /= 255
        mean = augimgs_train.mean(axis=0)
        augimgs_train -= mean
        augimgs_mask_train /= 255
        augimgs_mask_train[augimgs_mask_train > 0.5] = 1
        augimgs_mask_train[augimgs_mask_train <= 0.5] = 0
        return augimgs_train, augimgs_mask_train

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
if __name__ == '__main__':
    mydata = dataProcess(512, 512)
    mydata.create_train_data()
