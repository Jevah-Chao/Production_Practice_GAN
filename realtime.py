import cv2
import numpy as np
import os
# from test import test
from glob import glob
from utils import *
from net import generator
from matplotlib import pyplot as plt

def use_image_as_test(image, size):
    img = image.astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocessing(img,size)
    img = np.expand_dims(img, axis=0)
    return np.asarray(img)


def gan(style_name, sample, img_size=[256, 256]):
    test_data = use_image_as_test(sample, img_size)

    fake_img = sess.run(test_generated, feed_dict = {test_real : test_data})

    return fake_img


def get_gan(frame, size, style="S"):
    frame = cv2.resize(frame, size, interpolation=cv2.INTER_NEAREST)
    images = gan(style, frame, size)
    res = inverse_transform(images.squeeze())
    os.system("cls")
    res_bgr = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
    return res_bgr

def write_file():
    imgpath = "JapanStreet.jpg"
    frame = cv2.imread(imgpath)
    img = cv2.imread(imgpath)
    
    frame = get_gan(frame, size=(480, 218), style="H")
    
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    plt.figure()
    title1 = "AnimeGAN Generated"
    plt.subplot(1, 2, 1)
    plt.imshow(frame_bgr)
    plt.title(title1)
    title2 = "Origin"
    plt.subplot(1, 2, 2)
    plt.imshow(img_bgr)
    plt.title(title2)
    plt.show()
    
    cv2.imwrite('Gan_resultH.png', frame)

def start_capture():
     cap = cv2.VideoCapture(0)
     cap.set(3, 960)
     cap.set(4, 540)

     while True:
        # get a frame
        ret, frame = cap.read()
        
        # frame = get_gan(frame, size=(240, 109), style="S")
        frame = get_gan(frame, size=(480, 218), style="H")

        frame = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_NEAREST)

        # show a frame
        cv2.imshow("capture", frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
             break

     cap.release()
     cv2.destroyAllWindows()
    
    

if __name__ == '__main__':
    os.system("cls")
    test_real = tf.placeholder(tf.float32, [1, None, None, 3], name='test')
    with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
        test_generated = generator.G_net(test_real).fake
    saver = tf.train.Saver()
    gpu_options = tf.GPUOptions(allow_growth=True)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)) as sess:
        # load model
        ckpt = tf.train.get_checkpoint_state('checkpoint/')  # checkpoint file information
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)  # first line..
            saver.restore(sess, os.path.join('checkpoint/', ckpt_name))
            print(" [*] Success to read {}".format(ckpt_name))
            # print("[*] Wait Generating")
        else:
            print(" [*] Failed to find a checkpoint")
            exit(-1)

        # start_capture()
        write_file()

