#@silence 20170517
from unet import *
from data import dataProcess
import matplotlib.pyplot as plt
import numpy as np

# mydata = dataProcess(512,512)

imgs_test = np.load('imgs_test.npy')

# myunet = myUnet()
#
# model = myunet.get_unet()
#
# model.load_weights('unet.hdf5')
#
# imgs_mask_test = model.predict(imgs_test, verbose=1)
#
# np.save('imgs_mask_test.npy', imgs_mask_test)

imgs_test_predict = np.load('imgs_mask_test.npy')
print(imgs_test.shape, imgs_test_predict.shape)


n = 2
plt.figure(figsize=(20, 4))
for i in range(20, 22):
    plt.gray()
    ax = plt.subplot(2, n, (i-20)+1)
    plt.imshow(imgs_test[i].reshape(512, 512))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, (i - 20) + n + 1)
    plt.imshow(imgs_test_predict[i].reshape(512, 512))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
