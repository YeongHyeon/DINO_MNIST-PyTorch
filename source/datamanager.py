import random
import numpy as np
import source.utils as utils
import tensorflow as tf
from sklearn.utils import shuffle
from scipy.ndimage import gaussian_filter
from scipy.ndimage.interpolation import rotate

class DataSet(object):

    def __init__(self):

        print("\nInitializing Dataset...")
        self.num_class = 10
        self.__reset_index()
        self.__preparing()
        self.__reset_index()

    def __reset_index(self):

        self.idx_tr, self.idx_val, self.idx_te = 0, 0, 0

    def __preparing(self):
        (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.mnist.load_data()
        self.x_tr, self.y_tr = x_tr, y_tr
        self.x_tr, self.y_tr = shuffle(self.x_tr, self.y_tr)

        num_val = int(self.x_tr.shape[0] * 0.1)
        self.x_val, self.y_val = self.x_tr[:num_val], self.y_tr[:num_val]
        self.x_tr, self.y_tr = self.x_tr[num_val:], self.y_tr[num_val:]
        self.x_te, self.y_te = x_te, y_te

        ftxt = open("training_summary.txt", "w")
        for i in range(10):
            text = "Class-%d: %5d samples" %(i, np.sum(self.y_tr == i))
            print(text)
            ftxt.write("%s\n" %(text))
        ftxt.close()

        self.num_tr, self.num_val, self.num_te = self.x_tr.shape[0], self.x_val.shape[0], self.x_te.shape[0]
        dict_tmp = self.next_batch(ttv=0)
        x_tmp = dict_tmp['x1']
        self.dim_h, self.dim_w, self.dim_c = x_tmp.shape[1], x_tmp.shape[2], 1
        self.num_class = 10

    def augmentation(self, x):

        x = rotate(x, random.uniform(-10, 10), reshape=False, mode='nearest')
        x = np.roll(x, shift=int(random.uniform(-(x.shape[0]/20), x.shape[0]/20)), axis=0)
        x = np.roll(x, shift=int(random.uniform(-(x.shape[0]/20), x.shape[0]/20)), axis=1)
        x = gaussian_filter(x, sigma=1)

        return x

    def next_batch(self, batch_size=1, ttv=0):

        if(ttv == 0):
            idx_d, num_d, data, label = self.idx_tr, self.num_tr, self.x_tr, self.y_tr
        elif(ttv == 1):
            idx_d, num_d, data, label = self.idx_te, self.num_te, self.x_te, self.y_te
        else:
            idx_d, num_d, data, label = self.idx_val, self.num_val, self.x_val, self.y_val

        batch_x1, batch_x2, batch_y, terminate = [], [], [], False
        while(True):
            try:
                tmp_x = np.expand_dims(utils.min_max_norm(data[idx_d]), axis=-1)
                if(ttv == 0):
                    tmp_x1 = self.augmentation(tmp_x.copy())
                    tmp_x2 = self.augmentation(tmp_x.copy())
                else:
                    tmp_x1 = tmp_x.copy()
                    tmp_x2 = tmp_x.copy()
                tmp_y = label[idx_d]
            except:
                idx_d = 0
                if(ttv == 0):
                    self.x_tr, self.y_tr = shuffle(self.x_tr, self.y_tr)
                terminate = True
                break

            batch_x1.append(tmp_x1)
            batch_x2.append(tmp_x2)
            batch_y.append(tmp_y)
            idx_d += 1

            if(len(batch_x1) >= batch_size): break

        batch_x1 = np.asarray(batch_x1)
        batch_x2 = np.asarray(batch_x2)
        batch_y = np.asarray(batch_y)

        if(ttv == 0): self.idx_tr = idx_d
        elif(ttv == 1): self.idx_te = idx_d
        else: self.idx_val = idx_d

        return {'x1':batch_x1.astype(np.float32), 'x2':batch_x2.astype(np.float32), 'y':batch_y.astype(np.float32), 'terminate':terminate}
