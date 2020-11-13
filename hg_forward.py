import tensorflow as tf 
import numpy as np
import cv2
import os
#from inference import Inference
import pickle
import time as t

class HG_Forward_tiny():

    def __init__(self,pic_name = '2.jpg'):
        
        with open('./all_weights_cbr', 'rb') as fc:
            self.W = pickle.load(fc)
        self.img = cv2.imread(pic_name)
        self.round = 0
        self.cam_res = (480,640)
        self.nStack = 2
        self.nFeat = 2
        self.plt_j = True
        

    #重要参数
        #判断阈值设置 默认初始值：0.2
        self.thresh = 0.2
        

    #以下是预测初始化
        #颜色初始化

        #点以及线的颜色
        self.color = [(241,242,224), (196,203,128), (136,150,0), (64,77,0), 
				(201,230,200), (132,199,129), (71,160,67), (32,94,27),
				(130,224,255), (7,193,255), (0,160,255), (0,111,255),
				(220,216,207), (174,164,144), (139,125,96), (100,90,69),
				(252,229,179), (247,195,79), (229,155,3), (155,87,1),
				(231,190,225), (200,104,186), (176,39,156), (162,31,123),
				(210,205,255), (115,115,229), (80,83,239), (40,40,198)]
		# Color Names
        self.color_name = ['teal01', 'teal02', 'teal03', 'teal04',
				'green01', 'green02', 'green03', 'green04',
				'amber01', 'amber02', 'amber03', 'amber04',
				'bluegrey01', 'bluegrey02', 'bluegrey03', 'bluegrey04',
				'lightblue01', 'lightblue02', 'lightblue03', 'lightblue04',
				'purple01', 'purple02', 'purple03', 'purple04',
				'red01', 'red02', 'red03', 'red04']
        self.classes_name =  ["aeroplane", "bicycle", "bird",
				"boat", "bottle", "bus", "car", "cat", "chair",
				"cow", "diningtable", "dog", "horse", "motorbike",
				"person", "pottedplant", "sheep",
				"sofa", "train","tvmonitor"]
		# Person ID = 14 （YOLO标签？）
        self.color_class = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
        self.palette = {}	#{color name:color data}
        for i, name in enumerate(self.color_name):	#enumerate函数返回值：元素索引值+列表元素
            self.palette[name] = self.color[i]

        #连接线初始化
        self.links = {}	#{索引值:{'link':link data,'color':}}
		# Edit Links with your needed skeleton
		#节点之间的连线
        LINKS = [(0,1),(1,2),(2,6),(6,3),(3,4),(4,5),(6,8),(8,13),(13,14),(14,15),(8,12),(12,11),(11,10)]
        self.LINKS_ACP = [(0,1),(1,2),(3,4),(4,5),(7,8),(8,9),(10,11),(11,12)]
        color_id = [1,2,3,3,2,1,5,27,26,25,27,26,25]	#设置各连线的颜色ID
        self.color_id_acp = [8,9,9,8,19,20,20,19]
        for i in range(len(LINKS)):
            #字典里面套嵌字典，设定各个连线的颜色
            self.links[i] = {'link' : LINKS[i], 'color' : self.palette[self.color_name[color_id[i]]]}

    def test(self):
        for i in range(5):
            a = self._get_weights()
            print(i)
            print(a[0].shape)

    def givePixel(self, link, joints):
        return (joints[link[0]].astype(np.int), joints[link[1]].astype(np.int))

    def _get_weights(self):
        print('extract at : ' + str(self.round))
        s = 'r_' + str(self.round)
        www = self.W[s]
        print('conv_weights shape : ' + str(www[0].shape))
        self.round += 1
        return www



    def _conv_bn_relu(self, inputs,strides = 1, pad = 'VALID'):
        WW = self._get_weights()
        #pad = tf.pad(inputs, np.array([[0,0],[1,1],[1,1],[0,0]]))
        conv = tf.nn.conv2d(inputs, WW[0], [1,strides,strides,1], padding='VALID', data_format='NHWC')
        conv += WW[1]
        print('After conv+bn ' + ' conv result shape is ' + str(conv.shape))
        print()
        
        return tf.nn.relu(conv) 

    def _conv_block(self, inputs):
        WW = self._get_weights()
        pad = tf.pad(inputs, np.array([[0,0],[1,1],[1,1],[0,0]]))
        # if self.round == 6:
        #     print('sssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss')
        #     padss = pad.eval()
        #     padss.astype('float16').tofile("map_in_34_34.data")  
        #     print(padss.shape)  
              
        conv = tf.nn.conv2d(pad, WW[0], [1,1,1,1], padding='VALID', data_format='NHWC')

        #ss = conv.eval()
        #print(ss)
        print('After conv at '  + ' conv result shape is ' + str(conv.shape))
        conv += WW[1]
        
        return tf.nn.relu(conv)



    def _residual(self, inputs):    
        print()
        print('-'*5 + '  residual start  ' + '-'*5)
        conv = self._conv_block(inputs)
        print('residual input shape : ' + str(inputs.shape))
        print('residual conv3 shape : ' + str(conv.shape))
        if inputs.get_shape().as_list()[3] == conv.get_shape().as_list()[3]:

            print('-'*5 + '  residual end  ' + '-'*5)
            print()
            return tf.add_n([conv, inputs])
        else:
            W = self._get_weights()
            #pad = tf.pad(inputs, np.array([[0,0],[1,1],[1,1],[0,0]]))
            convb = tf.nn.conv2d(inputs, W[0], [1,1,1,1], padding='VALID', data_format='NHWC')
            print('residual conv1 shape : ' + str(convb.shape))
            print('After conv at '  + ' conv result shape is ' + str(convb.shape))
            
            print('-'*5 + '  residual end  ' + '-'*5)
            print()
            return tf.add_n([conv, convb])


    def _get_img(self):
        """ 
        Returns:
            self.img_res
            selg.img_hg
        """
        img = self.img
        img = cv2.flip(img, 1)	#图像水平翻转
        img[:, self.cam_res[1]//2 - self.cam_res[0]//2:self.cam_res[1]//2 + self.cam_res[0]//2]	#切片是左闭右开 ，高不变，宽变为(80:560)
        self.img_res = cv2.resize(img, (400,400))
        img_hg = cv2.resize(img, (256,256))
        img_hg = cv2.cvtColor(img_hg, cv2.COLOR_BGR2RGB)
        img_hg = np.expand_dims(img_hg/255,axis = 0)
        self.img_hg = img_hg.astype(np.float16)
        print(img_hg.dtype)
        print('input image shape : ' + str(img_hg.shape))

    def _preprocessing(self):
        """
        Return:
            r3
        """
        pad1 = np.lib.pad(self.img_hg,((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
        #print(pad1)
        print('padding shape : ' + str(pad1.shape))
        
        conv1 = self._conv_bn_relu(pad1,strides=2)

        print('--'*5 + '  residual start  ' + '--'*5)
        r1 = self._residual(conv1)

        print('after residual -- r1 shape : ' + str(r1.shape))
        print('--'*5 + '  residual over   -' + '--'*5)
        print()
        pool1 = tf.contrib.layers.max_pool2d(r1, [2,2], [2,2], padding='VALID')
        #print('after max pooling -- pool1 shape : ' + str(pool1.shape))

        r3 = self._residual(pool1)

        return r3
        #进入hourglass
        # Storage Table

    def _hourglass(self, inputs, n):
        up_1 = self._residual(inputs)
        low_ = tf.contrib.layers.max_pool2d(inputs, [2,2], [2,2], padding='VALID')
        print('after max pooling -- low_ shape : ' + str(low_.shape))
        low_1 = self._residual(low_)
        if n > 0:
            low_2 = self._hourglass(low_1, n-1)
            print('start')
        else:
            low_2 = self._residual(low_1)

        low_3 = self._residual(low_2)
        up_2 = tf.image.resize_nearest_neighbor(low_3, tf.shape(low_3)[1:3]*2)
        return tf.add_n([up_2,up_1])

    def _predict(self, hg):
        j = np.ones(shape = (16,2)) * -1
        #hg = hg.astype(np.float32)
        for i in range(len(j)):
                idx = np.unravel_index( hg[0,:,:,i].argmax(), (64,64))	#得到第i个节点中概率最大的那个点在图像中的索引值（在像素为64*64图像中的位置）
                print(idx)
                print(hg[0, idx[0], idx[1], i])
                if hg[0, idx[0], idx[1], i] > self.thresh:	#如果该点的概率大于阈值
                    j[i] = np.asarray(idx) * 400 / 64	#得到i节点在原图中的坐标
                    if self.plt_j:
                        cv2.circle(self.img_res, center = tuple(j[i].astype(np.int))[::-1], radius= 5, color= self.color[i][::-1], thickness= -1)
        for i in range(len(self.links)):
                    l = self.links[i]['link']
                    good_link = True
                    for p in l:
                        if np.array_equal(j[p], [-1,-1]):
                            good_link = False
                    if good_link:
                        pos = self.givePixel(l, j)
                        cv2.line(self.img_res, tuple(pos[0])[::-1], tuple(pos[1])[::-1], self.links[i]['color'][::-1], thickness = 5)
        cv2.imshow('stream', self.img_res)
        cv2.waitKey(0)

    def HG(self):
        print('start!')
        all_start = t.time()
        with tf.Session():
    
        #开始读取图片
            
            self._get_img()
        #预处理
            pre_s = t.time()
            _hourglass_input = self._preprocessing()
            pre_e = t.time()
            
            hg = [None] * self.nStack
            ll = [None] * self.nStack
            #ll_ = [None] * self.nStack
            #drop = [None] * self.nStack
            out = [None] * self.nStack
            out_ = [None] * self.nStack
            sum_ = [None] * self.nStack

            hg[0] = self._hourglass(_hourglass_input, self.nFeat)
            
            print(hg[0])
            
            #跳过了dropout层，将shape[0]改成了hg[0]
            #drop[0] = tf.layers.dropout(hg[0], rate = self.dropout_rate, training = self.training, name = 'dropout')
            ll[0] = self._conv_bn_relu(hg[0])
            w = self. _get_weights()
            out[0] = tf.nn.conv2d(ll[0],w[0], [1,1,1,1], padding='VALID', data_format='NHWC')
            print('After conv at ' + str(self.round) + ' conv result shape is ' + str(out[0].shape))
            print()
            w = self. _get_weights()
            out_[0] = tf.nn.conv2d(out[0], w[0], [1,1,1,1], padding='VALID', data_format='NHWC')
            print('After conv at ' + str(self.round) + ' conv result shape is ' + str(out_[0].shape))
            print()
            sum_[0] = tf.add_n([out_[0], ll[0], _hourglass_input])

            
            hg[1] = self._hourglass(sum_[0], self.nFeat)
            print('almost there !!!!!')
            #跳过了dropout层，将shape[1]改成了hg[1]
            #drop[1] = tf.layers.dropout(hg[self.nStack-1], rate = self.dropout_rate, training = self.training, name = 'dropout')
            ll[1] = self._conv_bn_relu(hg[1])
            w = self. _get_weights()
            out[1] = tf.nn.conv2d(ll[1], w[0], [1,1,1,1], padding='VALID', data_format='NHWC')
            print('After conv at ' + str(self.round) + ' conv result shape is ' + str(out[1].shape))
            print()

            print(out)
            print('finally !!!!!')
            stack = tf.stack(out, axis= 1)
            print(stack.shape)
        #绘图
            #对结果进行sigmoid
            hgg = tf.nn.sigmoid(stack[:,self.nStack - 1])

            with tf.Session():
                out____ = out[0].eval()
                hg = hgg.eval()
                print(hg[0][0][0][0])
            hg_e = t.time()
            print('Prerpocessing_end :')
            print(pre_e-pre_s)
            print('HG_end :')
            print(hg_e-pre_e)
            print('Model_end :')
            print(hg_e-all_start)
            print()
            self._predict(hg)

#pred = Inference(model='./weight/hg_refined_200_195')
#pred.webcamSingle()
#pred.webcamImage()

pred_ = HG_Forward_tiny()
pred_.HG()




