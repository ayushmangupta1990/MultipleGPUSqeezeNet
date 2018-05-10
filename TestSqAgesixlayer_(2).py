import time
t0 = time.time()
import numpy as np
import itertools, textwrap
import math, csv
import os, glob
import random
import cv2, numpy
import tensorflow as tf
import Modelsixlayer #importing the Model script
import os,sys,warnings,base64,json,math
from random import randint
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from collections import OrderedDict
from random import randint
from sklearn.metrics import confusion_matrix


record_csv = True


t0 = time.time()
DIGITS = "456790"
LETTERS = "BFHJKLMNPRTVWX"
CHARS = "012"


def unzip(b):
    xs, ys = zip(*b)
    xs = np.array(xs)
    ys = np.array(ys)
    return xs, ys


def code_to_vec(code):
    def char_to_vec(c):
        y = np.zeros((len(CHARS),))
        y[CHARS.index(c)] = 1.0
        return y
    c = np.vstack([char_to_vec(c) for c in code])
    return c.flatten()


def vec_to_plate(v):
    return "".join(CHARS[i] for i in v)


# def best_channel(im):
#     img = np.array(im)
#     a = np.argmax([np.mean(img[:,:,0]),np.mean(img[:,:,1]),np.mean(img[:,:,2])])
#     return Image.fromarray(img[:,:,a])


def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def, 
            input_map=None, 
            return_elements=None, 
            name="prefix", 
            op_dict=None, 
            producer_op_list=None)
    return graph


####


def report():
    x, y, params = Modelsixlayer.build_model_f()
    y_ = tf.placeholder(tf.float32, [None, 1 * len(CHARS)])
    best = tf.argmax(tf.reshape(y, [-1, 1, len(CHARS)]), 2)
    correct = tf.argmax(tf.reshape(y_, [-1, 1, len(CHARS)]), 2)
    
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    
    
    
    
    #f = np.load('Model/AgeWeightWestAsianAllEthnic12Mar.npz')
    for wt in glob.glob('Model_SavedWts/*.npz'):

        # f = np.load("jeansLength_2900.npz")
        # #f = np.load('BC_Weights_OG.npz')
        # initial_weights = [f[p] for p in sorted(f.files,key=lambda s: int(s[4:]))]
        # assign_ops = [w.assign(v) for w, v in zip(params, initial_weights)] 
        # sess.run(assign_ops)
        
        # from tensorflow.python.framework.graph_util import convert_variables_to_constants
        # minimal_graph = convert_variables_to_constants(sess, sess.graph_def, ['output_node'])
        # tf.train.write_graph(minimal_graph, 'ModelAllTest', 'JeansT.pb', as_text=False)
        
        Sq_graph = load_graph("/ayu-disk/MGPU/ckpt/jeans1.pb")
        Sq_input = Sq_graph.get_tensor_by_name('prefix/input_node:0')
        Sq_output = Sq_graph.get_tensor_by_name('prefix/squeezenet/output_node:0')
        s = tf.Session(graph=Sq_graph); 
        
        Matches1 = []; Matches2 = []
        Orig1 = []; Orig2 = []
        Pred1 = []; Pred2 = []
        
        check1off=[]
        

        checkimage=[]

        countMatches=0
        countoneoff=0




        OrigEth=[]

        #for f in (glob.glob("EthnicData/TestData/OrigCaucasianNearlyQC/*")):
        #for f in (glob.glob("/home/ayushman/AP/dataDownload/all_data/*")):
        for f in (glob.glob("/ayu-disk/sqJens/dataDownload/test_data/*")):
        
            
        #for f in (glob.glob("moreimages/*")):
            #try: 
                im = cv2.imread(f)
                img=im
                im = cv2.resize(im, (256,256))
                im1 = np.array(im)/ 255.
                code = f.split('/')[-1][:1]
                ctv = code_to_vec(code)
                p = s.run(Sq_output,feed_dict={Sq_input: im1.reshape([1,256,256,3])})
                pp = np.argmax(p.reshape([1,3]), 1); qq = np.argmax(ctv.reshape([1,3]), 1)
                
                Matches1.append(np.sum(pp == qq)); Orig1.append(vec_to_plate(qq)); Pred1.append(vec_to_plate(pp))
                if (pp!=qq):
                    print(f,pp,qq)
                

                actualcat=f.split('/')[-1][:1]
                OrigEth.append(actualcat)

                if(vec_to_plate(qq)==vec_to_plate(pp)):
                    countMatches=countMatches+1
                elif(int(vec_to_plate(pp))+1==int(vec_to_plate(qq)) or int(vec_to_plate(pp))-1==int(vec_to_plate(qq))):
                    countoneoff=countoneoff+1
                    check1off.append([f,vec_to_plate(qq),vec_to_plate(pp)])
                    #cv2.imwrite("oneoffinTestCopy/"+f.split('/')[-1],img)

                else:
                    checkimage.append([f,vec_to_plate(qq),vec_to_plate(pp)])
                        #cv2.imwrite("wronginTestCopy/"+f.split('/')[-1],img)

                # print(" ")       
                # print("Original:",Orig1)
                # print("Predicted:", Pred1)
                # print("Matches:",Matches1)
                # print(" ") 
                # print("check 1 off :") 
                # check1off.sort()
                # print('\n'.join('{}: {}'.format(*k) for k in enumerate(check1off)))
                # print(" ") 
                # print("image check:")
                # checkimage.sort()
                # print('\n'.join('{}: {}'.format(*k) for k in enumerate(checkimage)))

                # print("countMatches",countMatches)
                # print("Count one Off :",countoneoff)
                # print("count Total one off:",countoneoff+countMatches)

                #print("total test examples",len(Matches1))



                    #except:
                    #    print f.split('/')[-1][:-4]
                    

                
                #print ''
        char_acc1=100*(float(countMatches)/float(len(Matches1)))
        char_acc1off=100*(float(countMatches+countoneoff)/float(len(Matches1)))

        print '########### Exact_Acc: {:2.02f}% Model No {}'.format(char_acc1,wt)
        # print '1 Off_Acc : {:2.02f}%'.format(char_acc1off)

        cm=confusion_matrix(Orig1, Pred1)
        print("")
        print("\nConfusion Matrix:")
        print(cm)
        # #cm2=100*(cm / cm.astype(np.float).sum(axis=1))
        # cm2 = 100*(np.true_divide(cm, cm.sum(axis=1, keepdims=True)))
        # print(cm2)


        # cmf=confusion_matrix(OrigEth, Orig1)
        # print(cmf)
        break
        

        
report()
