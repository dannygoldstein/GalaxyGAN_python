from config import Config as conf
from data import *
from PIL import Image
from model import CGAN
from utils import imsave
import tensorflow as tf
import numpy as np
import time
import imageio
import sys
import wandb
from collections import defaultdict
import cv2

def prepocess_train(img, cond,):
    
    img =np.array(Image.fromarray(img).resize([conf.adjust_size, conf.adjust_size]))
    cond =np.array(Image.fromarray(cond).resize([conf.adjust_size, conf.adjust_size]))
    
    #h1 = int(np.ceil(np.random.uniform(1e-2, conf.adjust_size - conf.train_size)))
    #w1 = int(np.ceil(np.random.uniform(1e-2, conf.adjust_size - conf.train_size)))
    #img = img[h1:h1 + conf.train_size, w1:w1 + conf.train_size]
    #cond = cond[h1:h1 + conf.train_size, w1:w1 + conf.train_size]
    
    if np.random.random() > 0.5:
        img = np.fliplr(img)
        cond = np.fliplr(cond)
    img = img/127.5 - 1.
    cond = cond/127.5 - 1.
    img = img.reshape(1, conf.img_size, conf.img_size, conf.img_channel)
    cond = cond.reshape(1, conf.img_size, conf.img_size, conf.img_channel)
    return img,cond

def prepocess_test(img, cond):

    img =np.array(Image.fromarray(img).resize([conf.train_size, conf.train_size]))
    cond =np.array(Image.fromarray(cond).resize([conf.train_size, conf.train_size]))
    img = img.reshape(1, conf.img_size, conf.img_size, conf.img_channel)
    cond = cond.reshape(1, conf.img_size, conf.img_size, conf.img_channel)
    img = img/127.5 - 1.
    cond = cond/127.5 - 1.
    return img,cond

    
def train(model, wandb_api_key=None):
    
    data = load_data()
    images_for_table = defaultdict(list)
    wandb.login(key=wandb_api_key)

    d_opt = tf.train.AdamOptimizer(learning_rate=conf.learning_rate).minimize(model.d_loss, var_list=model.d_vars)
    g_opt = tf.train.AdamOptimizer(learning_rate=conf.learning_rate).minimize(model.g_loss, var_list=model.g_vars)

    saver = tf.train.Saver()

    start_time = time.time()
    if not os.path.exists(conf.data_path + "/checkpoint"):
        os.makedirs(conf.data_path + "/checkpoint")
    if not os.path.exists(conf.output_path):
        os.makedirs(conf.output_path)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    wandb.config = {key: value for key, value in conf.__dict__.items() if not key.startswith('_')}

    #frameSize = (conf.img_size * 3, conf.img_size)

    writers = []

    test_data = data["test"]()

    names = []
    for i, (img, cond, name) in zip(range(conf.n_test_save or 100), test_data):
        vidname = f'{name}.mp4'
        names.append(vidname)

    wandb.init(project='GalaxyGAN')
    test_evolution = wandb.Artifact(f'test_evolution_{wandb.run.id}', type='predictions')
    columns = ['id'] + [f'epoch{i}' for i in range(conf.max_epoch)]    
    table = wandb.Table(columns=columns)

    with tf.Session(config=config) as sess:
        if conf.model_path_train == "":
            sess.run(tf.global_variables_initializer())
        else:
            saver.restore(sess, conf.model_path_train)
            
        counter = 0
        for epoch in range(conf.max_epoch):
            train_data = data["train"]()
            
            for i, (img, cond, name) in enumerate(train_data):

                if not i < conf.n_train:
                    break
                
                img, cond = prepocess_train(img, cond)
                _, m = sess.run([d_opt, model.d_loss], feed_dict={model.image:img, model.cond:cond})
                #_, m = sess.run([d_opt, model.d_loss], feed_dict={model.image:img, model.cond:cond})
                _, M = sess.run([g_opt, model.g_loss], feed_dict={model.image:img, model.cond:cond})
                counter += 1
                if counter % 10 ==0:
                    print("Epoch [%d], Iteration [%d]: time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, counter % 4001, time.time() - start_time, m, M))
                    wandb.log({'generator_loss': M, 'discriminator_loss': m})
                    
                    
            if (epoch + 1) % conf.save_per_epoch == 0:
                save_path = saver.save(sess, conf.data_path + "/checkpoint/" + "model_%d.ckpt" % (epoch+1))
                print("Model saved in file: %s" % save_path)
                test_data = data["test"]()
                images = []
                for idx, (img, cond, name) in enumerate(test_data):
                    pimg, pcond = prepocess_test(img, cond)
                    gen_img = sess.run(model.gen_img, feed_dict={model.image:pimg, model.cond:pcond})
                    gen_img = gen_img.reshape(gen_img.shape[1:])
                    gen_img = (gen_img + 1.) * 127.5
                    image = np.concatenate((gen_img, cond, img), axis=1).astype(np.uint8)
                    #imsave(image, conf.output_path + "/%s" % name)

                    if idx < (conf.n_test_save or 100):
                        wandb_image = wandb.Image(image, caption=f'{name}_epoch{epoch}')
                        images.append(wandb_image)
                        images_for_table[idx].append(image)
                    else:
                        break
                        
                wandb.log({f'predictions_epoch{epoch}': images})


    for i, name in enumerate(names):
        table.add_data(name, *[wandb.Image(data) for data in images_for_table[i]])
        imageio.mimsave(name, images_for_table[i], format='GIF', duration=len(images_for_table[i]) / 3.)
                        
    wandb.log({'videos': [wandb.Video(name) for name in names]})
    test_evolution.add(table, 'training_evolution')
    wandb.run.log_artifact(test_evolution)
                    

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'gpu=':
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"]=str(sys.argv[1][4:])
    else:
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"]="1"
    train()
