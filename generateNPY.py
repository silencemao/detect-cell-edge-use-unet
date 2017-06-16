#@silence 20170517
from keras.preprocessing.image import load_img, img_to_array
import glob
import numpy as np
import os

path = '../deform/aug_train/0'         # 你的文件夹路径（一类的）
files = glob.glob(path + '/' + '*')    # 获得所有文件名字

imgdatas = np.ndarray((len(files), 512, 512, 1), dtype=np.uint8)    # 创建存储图像的空间　
imglabels = np.ndarray((len(files), 1))                             # 存储label
for i, file in enumerate(files):
    label = path[path.rindex('/') + 1:]                             # 获得label,此处为０
    img = load_img(file, grayscale=True)                            # 加载图像
    img = img_to_array(img)                                         # 转换为矩阵
    imgdatas[i] = img                                               # 当前图像矩阵放入列表中
    imglabels[i] = label                                            # 同上

np.save('train_img.npy', imgdatas)                                  # 保存文件到本文件夹下
np.save('train_label.npy', imglabels)


def create_train_data(self):
    i = 0
    count = 0                                           # 统计一共有多少图像
    for indir in os.listdir(self.aug_merge_path):
        path = os.path.join(self.aug_merge_path, indir)
        count += len(os.listdir(path))                  # 获得一个文件夹下的图像
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
    np.save(self.npy_path + '/augimgs_train.npy', imgdatas)  # 将30张训练集和30张label生成npy数据
    np.save(self.npy_path + '/augimgs_mask_train.npy', imglabels)
    print('Saving to .npy files done.')




