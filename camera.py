import cv2
import numpy as np
import os
from test import test, check_folder
from glob import glob
from utils import *
from net import generator


def gan(checkpoint_dir, style_name, test_dir, if_adjust_brightness, img_size=[256,256]):
    # tf.reset_default_graph()
    result_dir = 'results/'+style_name
    check_folder(result_dir)
    test_files = glob('{}/*.*'.format(test_dir))

    # test_real = tf.placeholder(tf.float32, [1, 256, 256, 3], name='test')
    test_real = tf.placeholder(tf.float32, [1, None, None, 3], name='test')

    with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
        test_generated = generator.G_net(test_real).fake
    saver = tf.train.Saver()

    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)) as sess:
        # tf.global_variables_initializer().run()
        # load model
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)  # checkpoint file information
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)  # first line
            saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Success to read {}".format(ckpt_name))
        else:
            print(" [*] Failed to find a checkpoint")
            return

        sample_file = test_files[0]

        sample_image = np.asarray(load_test_data(sample_file, img_size))
        image_path = os.path.join(result_dir,'{0}'.format(os.path.basename(sample_file)))
        fake_img = sess.run(test_generated, feed_dict = {test_real : sample_image})
        return fake_img


def run_gan(frame):
	cv2.imwrite("./dataset/test/temp/temp.png", frame)
	test('checkpoint/', 'S', 'dataset/test/temp', True)
	os.system("cls")
	print("done")
	res = cv2.imread("./results/S/temp.png")
	while True:
		cv2.imshow("result", res)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			break
	return


def get_gan(frame, size=None):
	if size:
		cv2.resize(frame, size, interpolation=cv2.INTER_CUBIC)
	cv2.imwrite("./dataset/test/temp/temp.png", frame)
	# test('checkpoint/', 'S', 'dataset/test/temp', True)
	images = gan('checkpoint/', 'S', 'dataset/test/temp', True)
	res = inverse_transform(images.squeeze())
	os.system("cls")
	# res = cv2.imread("./results/S/temp.png")
	for y in range(len(res)):
		for x in range(len(res[y])):
			res[y][x] = [
				res[y][x][2],
				res[y][x][1],
				res[y][x][0]
			]
	return res


def get_gan_list(frameset):
	for i, f in enumerate(frameset):
		cv2.imwrite("./dataset/test/temp_list/temp_{}.png".format(i), f)
	test('checkpoint/', 'S', 'dataset/test/temp_list', True)
	os.system("cls")
	for i in range(len(frameset)):
		frameset[i] = cv2.imread("./results/S/temp_{}.png".format(i))
	t = 0
	while True:
		# show a frame
		cv2.imshow("frame_set", frameset[t])

		t = (t + 1) % len(frameset)

		if cv2.waitKey(20) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			break
	return


def start_capture():
	cap = cv2.VideoCapture(0)
	cap.set(3, 960)
	cap.set(4, 540)
	# cap.set(3, 192)
	# cap.set(4, 108)

	# frameset = []

	while True:
		# get a frame
		ret, frame = cap.read()

		# frame = get_gan(frame, size=(16, 9))

		# cv2.resize(frame, (1280, 960), interpolation=cv2.INTER_CUBIC)

		# frameset.append(frame)

		# if len(frameset) > 60:
		# 	frameset = frameset[len(frameset) - 60:]

		# show a frame
		cv2.imshow("capture", frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			cap.release()
			cv2.destroyAllWindows()
			run_gan(frame)
			# get_gan_list(frameset)
			break


if __name__ == '__main__':
	os.system("cls")
	start_capture()
