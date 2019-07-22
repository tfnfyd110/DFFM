import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import numpy as np
from keras.models import *
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from data_multi import *
import tensorflow as tf


def my_dice(y_true, y_pred, threshold=keras.variable(value=0.5)):
    y_pred = keras.cast(y_pred >= threshold, 'float32')
    TP = keras.sum(y_pred * y_true)
    FP = keras.sum(y_pred - y_pred * y_true)
    FN = keras.sum(y_true - y_pred * y_true)
    return (2 * TP / (FN + FP + 2 * TP))


class myUnet(object):

    def __init__(self, img_rows = 256, img_cols = 256):
        self.img_cols = img_cols
        self.img_rows = img_rows

    def load_data(self):

        mydata = DataProcess(self.img_rows, self.img_cols)
        imgs_train_mri, imgs_mask_train_mri = mydata.load_train_data_mri()
        imgs_val_mri, imgs_mask_val_mri = mydata.load_val_data_mri()
        imgs_train_ct, imgs_mask_train_ct = mydata.load_train_data_ct()
        imgs_val_ct, imgs_mask_val_ct = mydata.load_val_data_ct()
        return imgs_train_mri, imgs_mask_train_mri, imgs_val_mri, imgs_mask_val_mri, imgs_train_ct, imgs_mask_train_ct, imgs_val_ct, imgs_mask_val_ct

    def get_unet(self):

        inputs_mri = Input((self.img_rows, self.img_cols, 4), name = 'input_mri')
        conv1_mri = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs_mri)
        print("conv1_mri shape:", conv1_mri.shape)
        conv1_mri = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1_mri)
        print("conv1_mri shape:", conv1_mri.shape)
        pool1_mri = MaxPooling2D(pool_size=(2, 2))(conv1_mri)
        print("pool1_mri shape:", pool1_mri.shape)

        conv2_mri = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1_mri)
        print("conv2_mri shape:",conv2_mri.shape)
        conv2_mri = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2_mri)
        print("conv2_mri shape:",conv2_mri.shape)
        pool2_mri = MaxPooling2D(pool_size=(2, 2))(conv2_mri)
        print("pool2_mri shape:",pool2_mri.shape)

        conv3_mri = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2_mri)
        print("conv3_mri shape:",conv3_mri.shape)
        conv3_mri = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3_mri)
        print("conv3_mri shape:",conv3_mri.shape)
        pool3_mri = MaxPooling2D(pool_size=(2, 2))(conv3_mri)
        print("pool3_mri shape:",pool3_mri.shape)

        conv4_mri = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3_mri)
        conv4_mri = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4_mri)
        drop4_mri = Dropout(0.5)(conv4_mri)
        pool4_mri = MaxPooling2D(pool_size=(2, 2))(drop4_mri)

        conv5_mri = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4_mri)
        conv5_mri = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5_mri)
        drop5_mri = Dropout(0.5)(conv5_mri)

        up6_mri = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5_mri))
        merge6_mri = concatenate([drop4_mri, up6_mri], axis=3)
        conv6_mri = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6_mri)
        conv6_mri = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6_mri)

        up7_mri = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6_mri))
        merge7_mri = concatenate([conv3_mri,up7_mri],  axis = 3)
        conv7_mri = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7_mri)
        conv7_mri = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7_mri)

        up8_mri = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7_mri))
        merge8_mri = concatenate([conv2_mri,up8_mri], axis = 3)
        conv8_mri = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8_mri)
        conv8_mri = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8_mri)

        up9_mri = Conv2D(64, 2, activation='relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8_mri))
        merge9_mri = concatenate([conv1_mri,up9_mri], axis = 3)
        conv9_mri = Conv2D(64, 3, activation='relu', padding = 'same', kernel_initializer = 'he_normal')(merge9_mri)

        inputs_ct = Input((self.img_rows, self.img_cols, 1), name = 'input_ct')
        conv1_ct = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs_ct)
        print("conv1_ct shape:", conv1_ct.shape)
        conv1_ct = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1_ct)
        print("conv1_ct shape:", conv1_ct.shape)
        pool1_ct = MaxPooling2D(pool_size=(2, 2))(conv1_ct)
        print("pool1_ct shape:", pool1_ct.shape)

        conv2_ct = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1_ct)
        print("conv2_ct shape:",conv2_ct.shape)
        conv2_ct = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2_ct)
        print("conv2_ct shape:",conv2_ct.shape)
        pool2_ct = MaxPooling2D(pool_size=(2, 2))(conv2_ct)
        print("pool2_ct shape:",pool2_ct.shape)

        conv3_ct = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2_ct)
        print("conv3_ct shape:",conv3_ct.shape)
        conv3_ct = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3_ct)
        print("conv3_ct shape:",conv3_ct.shape)
        pool3_ct = MaxPooling2D(pool_size=(2, 2))(conv3_ct)
        print("pool3_ct shape:",pool3_ct.shape)

        conv4_ct = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3_ct)
        conv4_ct = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4_ct)
        drop4_ct = Dropout(0.5)(conv4_ct)
        pool4_ct = MaxPooling2D(pool_size=(2, 2))(drop4_ct)

        conv5_ct = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4_ct)
        conv5_ct = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5_ct)
        drop5_ct = Dropout(0.5)(conv5_ct)

        up6_ct = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5_ct))
        merge6_ct = concatenate([drop4_ct, up6_ct], axis=3)
        conv6_ct = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6_ct)
        conv6_ct = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6_ct)

        up7_ct = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6_ct))
        merge7_ct = concatenate([conv3_ct,up7_ct],  axis = 3)
        conv7_ct = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7_ct)
        conv7_ct = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7_ct)

        up8_ct = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7_ct))
        merge8_ct = concatenate([conv2_ct,up8_ct], axis = 3)
        conv8_ct = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8_ct)
        conv8_ct = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8_ct)

        up9_ct = Conv2D(64, 2, activation='relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8_ct))
        merge9_ct = concatenate([conv1_ct,up9_ct], axis = 3)
        conv9_ct = Conv2D(64, 3, activation='relu', padding = 'same', kernel_initializer = 'he_normal')(merge9_ct)
        merge1_ct_mri = concatenate([conv9_mri,conv9_ct], axis = 3)
    
        conv1_ct_mri = Conv2D(64, 3, activation='relu', padding = 'same', kernel_initializer = 'he_normal')(conv1_ct_mri)
        conv1_ct_mri = Conv2D(64, 3, activation='relu', padding = 'same', kernel_initializer = 'he_normal')(conv1_ct_mri)
        conv1_ct_mri = Conv2D(2, 3, activation='relu', padding = 'same', kernel_initializer = 'he_normal')(conv1_ct_mri)
        conv2_ct_mri = Conv2D(1, 1, activation='sigmoid', name = 'output')(conv1_ct_mri)
        model = Model(input=[inputs_ct, inputs_mri], output = conv2_ct_mri)
		
		
        def focal_loss(gamma=2, alpha=0.6):
            def focal_loss_fixed(y_true, y_pred):
                eps = 1e-12
                y_pred = keras.clip(y_pred, eps, 1.-eps)
                pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
                pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
                return -keras.sum(alpha * keras.pow(1. - pt_1, gamma) * keras.log(keras.epsilon() + pt_1))-keras.sum((1-alpha) * keras.pow(pt_0, gamma) * keras.log(1. - pt_0 + keras.epsilon()))
            return focal_loss_fixed

        model.compile(optimizer=Adam(lr = 1e-4), loss=[focal_loss(gamma=2, alpha=0.6)], metrics=[my_dice, 'accuracy'])

        return model

    def train(self):

        print("loading data")
        imgs_train_mri, imgs_mask_train_mri, imgs_val_mri, imgs_mask_val_mri, imgs_train_ct, imgs_mask_train_ct, imgs_val_ct, imgs_mask_val_ct = self.load_data()
        model = self.get_unet()
        # model.load_weights('unet4.hdf5')
        print("got unet")
        filepath = "DFFM_2_0.6-{epoch:02d}--{val_my_dice:.3f}.hdf5"
        model_checkpoint = ModelCheckpoint(filepath, monitor= 'val_loss',verbose=1, save_best_only=False, mode = 'min')
        print('Fitting model...')
        model.fit(x={'input_mri': imgs_train_mri, 'input_ct': imgs_train_ct}, y={'output': imgs_mask_train_mri}, batch_size=1, epochs=25, verbose=1, validation_data = ( {'input_mri': imgs_val_mri, 'input_ct': imgs_val_ct},{'output': imgs_mask_val_mri}),shuffle = True, callbacks=[model_checkpoint])
        print(model.evaluate(x = {'input_mri': imgs_val_mri, 'input_ct': imgs_val_ct},y = {'output': imgs_mask_val_mri},batch_size = 1))
        #print('predict test data')
        #imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
        #np.save('result/imgs_mask_fl_test.npy', imgs_mask_test)

    def save_img(self):

        print("array to image")
        imgs = np.load('result/imgs_mask_fl_test.npy')
        textfile = open('npydata/test_name.txt')
        line = textfile.readline()
        parts = line.split(',')
        for i in range(imgs.shape[0]):
            img = imgs[i]
            tmp = parts[i]
            name = tmp[2:-5]
            img = array_to_img(img)
            img.save("result/seg_img_fl/"+ name + ".png")
  
                             
if __name__ == '__main__':
    myunet = myUnet()
    myunet.train()
    #myunet.save_img()
        







