### Predict tracking results

import logging
import time
import tensorflow as tf
import os
import goturn_net

from PIL import Image, ImageDraw
import numpy as np

NUM_EPOCHS = 500
BATCH_SIZE = 1
WIDTH = 227
HEIGHT = 227

logfile = "test.log"
test_txt = "test_set.txt"
def load_train_test_set(train_file):
    '''
    return train_set or test_set
    example line in the file:
    <target_image_path>,<search_image_path>,<x1>,<y1>,<x2>,<y2>
    (<x1>,<y1>,<x2>,<y2> all relative to search image)
    '''
    ftrain = open(train_file, "r")
    trainlines = ftrain.read().splitlines()
    train_target = []
    train_search = []
    train_box = []
    for line in trainlines:
        #print(line)
        line = line.split(",")
        # remove too extreme cases
        # if (float(line[2]) < -0.3 or float(line[3]) < -0.3 or float(line[4]) > 1.2 or float(line[5]) > 1.2):
        #     continue
        train_target.append(line[0])
        train_search.append(line[1])
        box = [10*float(line[2]), 10*float(line[3]), 10*float(line[4]), 10*float(line[5])]
        train_box.append(box)
    ftrain.close()
    print("len:%d"%(len(train_target)))
    
    return [train_target, train_search, train_box]

def data_reader(input_queue):
    '''
    this function only reads the image from the queue
    '''
    search_img = tf.read_file(input_queue[0])
    target_img = tf.read_file(input_queue[1])

    search_tensor = tf.to_float(tf.image.decode_jpeg(search_img, channels = 3))
    search_tensor = tf.image.resize_images(search_tensor,[HEIGHT,WIDTH],
                            method=tf.image.ResizeMethod.BILINEAR)
    target_tensor = tf.to_float(tf.image.decode_jpeg(target_img, channels = 3))
    target_tensor = tf.image.resize_images(target_tensor,[HEIGHT,WIDTH],
                            method=tf.image.ResizeMethod.BILINEAR)
    box_tensor = input_queue[2]
    return [search_tensor, target_tensor, box_tensor]


def next_batch(input_queue):
    min_queue_examples = 128
    num_threads = 8
    [search_tensor, target_tensor, box_tensor] = data_reader(input_queue)
    [search_batch, target_batch, box_batch] = tf.train.batch(
        [search_tensor, target_tensor, box_tensor],
        batch_size=BATCH_SIZE,
        num_threads=num_threads,
        capacity=min_queue_examples + (num_threads+2)*BATCH_SIZE)
    return [search_batch, target_batch, box_batch]


def transfer_to_center(im_size, coord_corner, ):
    #coord_corner: [x0, y0, x1, y1]
    #im_size: [width, height]
    center_x = .5 * (coord_corner[2] + coord_corner[0])/im_size[0]
    center_y = .5 * (coord_corner[1] + coord_corner[3])/im_size[1]
    width = 1.*(coord_corner[2] - coord_corner[0])/im_size[0]
    height = 1.*(coord_corner[3] - coord_corner[1])/im_size[1]
    target_bbox_pos = [center_x, center_y, width, height]

     #crop images
    target_bbox_pos_corner_twice = [(target_bbox_pos[0] - target_bbox_pos[2])*im_size[0],
                                (target_bbox_pos[1] - target_bbox_pos[3])*im_size[1], 
                                (target_bbox_pos[0] + target_bbox_pos[2])*im_size[0],
                                (target_bbox_pos[1] + target_bbox_pos[3])*im_size[1]]

    target_bbox_pos_corner_twice[0] = max(0, int(target_bbox_pos_corner_twice[0]))
    target_bbox_pos_corner_twice[1] = max(0, int(target_bbox_pos_corner_twice[1]))
    target_bbox_pos_corner_twice[2] = min(im_size[0], int(target_bbox_pos_corner_twice[2]))
    target_bbox_pos_corner_twice[3] = min(im_size[1], int(target_bbox_pos_corner_twice[3]))


if __name__ == "__main__":
    target_img_path = '/home/xiaoshi/workspace/git_repos/GOTURN-Tensorflow/imgs/1.jpg'
    search_img_path = '/home/xiaoshi/workspace/git_repos/GOTURN-Tensorflow/imgs/3.jpg'

    target_im = Image.open(target_img_path)
    search_im = Image.open(search_img_path)
    im_size = target_im.size
    #target_bbox_pos = [0.289,0.383,0.155,0.083] # center form

    target_bbox_pos = transfer_to_center(im_size, [827,287,919,349])
    # load images
    

    


    target_patch = target_im.crop(target_bbox_pos_corner_twice).resize((WIDTH, HEIGHT))
    search_patch = search_im.crop(target_bbox_pos_corner_twice).resize((WIDTH, HEIGHT))

    target_patch_cp = target_patch.copy()
    search_patch_cp = search_patch.copy()
    
    target_patch = np.expand_dims(target_patch, 0)
    search_patch = np.expand_dims(search_patch, 0)


    tracknet = goturn_net.TRACKNET(BATCH_SIZE, train = False)
    tracknet.build()

    #print "finish build net "
    sess = tf.Session()
    init = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()
    sess.run(init)
    sess.run(init_local)
    #print "finish init net "
    #coord = tf.train.Coordinator()
    # start the threads
    #tf.train.start_queue_runners(sess=sess, coord=coord)

    ckpt_dir = "./checkpoints"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    #print "before load ckpt"
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        #feed_dict = {tracknet.image:cur_batch[0], tracknet.target:cur_batch[1]}
        saver = tf.train.Saver()
        saver.restore(sess, ckpt.model_checkpoint_path)
        #print "finish restore"

    feed_dict= {tracknet.image:search_patch ,  tracknet.target:target_patch}
    #print "start run net"
    fc4 = sess.run([tracknet.fc4], feed_dict=feed_dict)

    fc4 = fc4[0][0] /10.

    print "result ",fc4
    #print fc4/10.
    im2 = search_patch_cp
    draw = ImageDraw.Draw(im2, 'RGBA')
    draw.rectangle((fc4[0]*WIDTH, fc4[1]*HEIGHT, fc4[2]*WIDTH, fc4[3]*HEIGHT), fill=(0, 255, 0,100))
    #im2.show()
    target_patch_cp.save('3_ori.png','PNG')
    im2.save('3_res.png','PNG')


