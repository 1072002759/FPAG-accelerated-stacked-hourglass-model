import time as t
import numpy as np
import pickle
from conv_64out import Conv
#import tensorflow as tf
import cv2

class HG_Xilinx():
    def __init__(self,pic_name = '2.jpg'):
        
        with open('./all_weights_opt_convert2xilinx', 'rb') as fc:
            self.W = pickle.load(fc)

        #固定参数
        self.x = Conv()
        self.img = cv2.imread(pic_name)
        self.round = 0
        self.cam_res = (480,640)
        self.nStack = 2
        self.nFeat = 2
        self.plt_j = True

        #self.weights_input_266 = np.zeros((3,64,6,6), dtype='float16')
        #self.map_in_input_266 = np.zeros((3,68,68), dtype='float16')
        #self.result_266 = np.zeros((64,128,128), dtype='float16')

        #self.weights_input = np.zeros((128,32,3,3), dtype='float16')
        #self.map_in_input = np.zeros((128,34,34), dtype='float16')
        #self.result = np.zeros((32,32,32), dtype='float16')

        #重要参数
        #判断阈值设置 默认初始值：0.2
        self.thresh = 0.20
        

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
            
#         self.residual_64_64_result = np.zeros((128,64,64), dtype='float16')
#         self.residual_32_32_result = np.zeros((128,32,32), dtype='float16')
#         self.residual_16_16_result = np.zeros((128,16,16), dtype='float16')
#         self.residual_8_8_result = np.zeros((128,8,8), dtype='float16')

    def givePixel(self, link, joints):
        return (joints[link[0]].astype(np.int), joints[link[1]].astype(np.int))
     
    def _get_img(self):
        """ 
        Returns:
            self.img_res
            selg.img_hg
        """
        img = self.img
        img = cv2.flip(img, 1)	#图像水平翻转
        img[:, self.cam_res[1]//2 - self.cam_res[0]//2:self.cam_res[1]//2 + self.cam_res[0]//2]	#切片是左闭右开 ，高不变，宽变为(80:560)
        self.img_res = cv2.resize(img, (800,800))
        img_hg = cv2.resize(img, (256,256))
        img_hg = cv2.cvtColor(img_hg, cv2.COLOR_BGR2RGB)
        img_hg = img_hg.astype(np.float16)
        #调换维度(256,256,3) -> (3，256，256)
        temp = img_hg.swapaxes(1,2)
        img_hg = temp.swapaxes(0,1)
        self.img_hg = img_hg/255
        self.img_hg = self.img_hg.astype(np.float16)
        
#         print(self.img_hg.dtype)
#         print('input image shape : ' + str(img_hg.shape))
        
    def _get_video(self):
        """ 
        Returns:
            self.img_res
            selg.img_hg
        """
        pass
        
    def _residual_128_1_1_3_3(self, inputs):
        '''
        状态：√
        实现 redsidual 3*3 1*1 (pad+conv+relu)
        输入: conv1
        输出：r1
        输入尺寸: (64,128,128)
        输出尺寸: (128,128,128)
        卷积尺寸: (64,128,3,3)
        卷积尺寸: (64,128,1,1)
        输入有效: (64,32,32)
        输出有效: (64,32,32)
        阶段: 1，2
        '''  
        ####r1 = self._residual(conv1) input shape:(64,128,128),conv shape:(64,128,3,3),each valid shape:(32,32,32),output shape:(128,128,128)
        #准备数据
        r1_conv_3 = np.zeros((128,128,128), dtype='float16')
        weights_3 = self.W['r_1'][0]
        beta_3 = self.W['r_1'][1]
        r1_conv_1 = np.zeros((128,128,128), dtype='float16')
        weights_1 = self.W['r_2'][0]
        beta_1 = np.zeros((64), dtype='float16')
        
        #卷积1*1
        #！！1*1卷积，pad是在右边与下边pad两层0
        #卷积次数
        #map_in_1 = np.lib.pad(conv1,((0, 64), (0, 2), (0, 2)), 'constant')
        for c in range(2): 
            #拆分
            for i in range(4): 
                for j in range(4):
                    #卷积 each valid shape:(32,32,32)
                    map_in_temp = np.lib.pad(inputs[:, i*32:32+i*32, j*32:32+j*32],((0, 64), (0, 2), (0, 2)), 'constant')
                    r1_conv_1[c*64:(c+1)*64, i*32:(i+1)*32, j*32:(j+1)*32] = self.x.conv_32_32_without_relu(map_in_temp, weights_1[:,c*64:(c+1)*64,:,:], beta_1)
        #卷积3*3
        map_in_3 = np.lib.pad(inputs,((0, 64), (1, 1), (1, 1)), 'constant')
#         print(map_in_3.shape)
        #卷积次数
        for c in range(2):
            #拆分
            for i in range(4):
                for j in range(4):
                    #卷积 each valid shape:(32,32,32) = 
                    r1_conv_3[c*64:(c+1)*64, i*32:(i+1)*32, j*32:(j+1)*32] = self.x.conv_32_32(map_in_3[:, i*32:34+i*32, j*32:34+j*32], weights_3[:,c*64:(c+1)*64,:,:], beta_3[c*64:(c+1)*64])
        x = r1_conv_3+r1_conv_1
#         print('_residual_128_1_1_3_3')
#         for i in range(8):
#             print(x[-1-i][i][i])
        return r1_conv_3+r1_conv_1
    
    def _conv_bn_relu_260_(self, inputs):
        '''
        状态：√
        实现 conv_bn_relu
        输入: pad1
        输出：conv1
        输入尺寸: (3,260,260)
        输出尺寸: (64,128,128)
        卷积尺寸: (3,64,6,6)
        输入有效: (3,68,68)
        输出有效: (16,32,32)
        阶段: 0
        ''' 
        conv1 = np.zeros((64,128,128), dtype='float16')
        beta = self.W['r_0'][1]
        weights = self.W['r_0'][0]
        #卷积次数

        for i in range(4): 
            for j in range(4):
                #！！最终每次输出有效的数据只有(16,32,32)
                conv1[:, i*32:(i+1)*32, j*32:(j+1)*32] = self.x.conv_68_68(inputs[:, i*64:68+i*64, j*64:68+j*64], weights[:,0:64,:,:], beta[0:64])
#         print('_conv_bn_relu_260_')
#         for i in range(8):
#             print(conv1[-1-i][i][i])
        return conv1
        
    def _residual_64_64(self, inputs, _round):
        '''
        状态：√
        实现redsidual 3*3 (pad+conv+relu)
        输入: inputs
        步骤: round
        输入尺寸: (128,64,64)
        输出尺寸: (128,64,64)
        卷积尺寸: (128,128,3,3)
        输入有效: (128,34,34)
        输出有效: (32,32,32)
        '''
        
        #准备数据
        residual_64_64_result = np.zeros((128,64,64), dtype='float16')
        weights_3 = self.W[_round][0]
        beta_3 = self.W[_round][1]
        
        #卷积3*3
        map_in_3 = np.lib.pad(inputs,((0, 0), (1, 1), (1, 1)), 'constant')
        #卷积次数
        for c in range(2): 
            #拆分
            for i in range(2): 
                for j in range(2):
                    #卷积 each valid shape:(32,32,32) = 
                    input_temp = map_in_3[:, i*32:34+i*32, j*32:34+j*32]
                    input_weight = weights_3[:,c*64:(c+1)*64,:,:]
                    input_beta = beta_3[c*64:(c+1)*64]
                    residual_64_64_result[c*64:(c+1)*64, i*32:(i+1)*32, j*32:(j+1)*32] = self.x.conv_32_32(input_temp, input_weight, input_beta)
                    #ss = map_in_3(input_temp, input_weight, input_beta)
#         x = inputs + residual_64_64_result
#         print(_round)
#         for i in range(8):
#             print(x[-1-i][i][i])
        return inputs + residual_64_64_result

    def _residual_32_32(self, inputs, _round):
        '''
        状态：√
        实现redsidual 3*3 (pad+conv+relu)
        输入: inputs
        步骤: round
        输入尺寸: (128,32,32)
        输出尺寸: (128,32,32)
        卷积尺寸: (128,128,3,3)
        输入有效: (128,34,34)
        输出有效: (32,32,32)
        '''
        #准备数据
        residual_32_32_result = np.zeros((128,32,32), dtype='float16')
        weights_3 = self.W[_round][0]
        beta_3 = self.W[_round][1]
        #卷积3*3
        #pad 输入
        map_in_3 = np.lib.pad(inputs,((0, 0), (1, 1), (1, 1)), 'constant')
        #卷积次数
        for c in range(2): 
            #拆分
            #卷积 each valid shape:(32,32,32) = 
            residual_32_32_result[c*64:(c+1)*64, :, :] = self.x.conv_32_32(map_in_3, weights_3[:,c*64:(c+1)*64,:,:], beta_3[c*64:(c+1)*64])
#         x = inputs + residual_32_32_result
#         print(_round)
#         for i in range(8):
#             print(x[-1-i][i][i])
        return inputs + residual_32_32_result

    def _residual_16_16(self, inputs, _round):
        '''
        状态：√
        实现redsidual 3*3 (pad+conv+relu)
        输入: inputs
        步骤: round
        输入尺寸: (128,16,16)
        输出尺寸: (128,16,16)
        卷积尺寸: (128,128,3,3)
        输入有效: (128,18,18)
        输出有效: (32,16,16)
        '''
        residual_16_16_result = np.zeros((128,16,16), dtype='float16')
        weights_3 = self.W[_round][0]
        beta_3 = self.W[_round][1]
        #卷积3*3
        map_in_3 = np.lib.pad(inputs,((0, 0), (1, 17), (1, 17)), 'constant')
        #卷积次数
        for c in range(2): 
            #拆分
            #卷积 each valid shape:(32,16,16)
            #缓存结果过 temp 从中切出有效部分
            temp = self.x.conv_32_32(map_in_3, weights_3[:,c*64:(c+1)*64,:,:], beta_3[c*64:(c+1)*64])
            residual_16_16_result[c*64:(c+1)*64, :, :] = temp[:,0:16,0:16]
#         x = inputs + residual_16_16_result
#         print(_round)
#         for i in range(8):
#             print(x[-1-i][i][i])
        return inputs + residual_16_16_result

    def _residual_8_8(self, inputs, _round):
        '''
        状态：√
        实现redsidual 3*3 (pad+conv+relu)
        输入: inputs
        步骤: round
        输入尺寸: (128,8,8)
        输出尺寸: (128,8,8)
        卷积尺寸: (128,128,3,3)
        输入有效: (128,10,10)
        输出有效: (32,8,8)
        '''
        residual_8_8_result = np.zeros((128,8,8), dtype='float16')
        weights_3 = self.W[_round][0]
        beta_3 = self.W[_round][1]
        #卷积3*3
        map_in_3 = np.lib.pad(inputs,((0, 0), (1, 25), (1, 25)), 'constant')
        #卷积次数
        for c in range(2): 
            residual_8_8_result[c*64:(c+1)*64] = self.x.conv_8_8(map_in_3, weights_3[:,c*64:(c+1)*64,:,:], beta_3[c*64:(c+1)*64])
#         x = inputs + residual_8_8_result
#         print(_round)
#         for i in range(8):
#             print(x[-1-i][i][i])
        return inputs + residual_8_8_result

    def _preprocessing(self):
        """
        Return:
            r3
        """
        pad1 = np.lib.pad(self.img_hg,((0, 0), (2, 2), (2, 2)), 'constant')
#         print(pad1)
#         print('padding shape : ' + str(pad1.shape))
            
        conv1 = self._conv_bn_relu_260_(pad1)

#         print('conv1')
#         for i in range(10):
#             print(conv1[-1-i][i][i])

        r1 = self._residual_128_1_1_3_3(conv1)
#         print('r1')
#         for i in range(10):
#             print(r1[-1-i][i][i])

#         print('after residual -- r1 shape : ' + str(r1.shape))
#         print('--'*5 + '  residual over   -' + '--'*5)
#         print()
        '''
        状态：待验证
        实现 Maxpooling 2*2
        输入: r1
        输出：pool1
        输入尺寸: (128,128,128)
        输出尺寸: (128,64,64)
        池化尺寸: (2，2)
        输入有效: (32,32,32)
        输出有效: (32,16,16)
        '''     
#         print('pool input')
#         for c in range(1):
#             for i in range(6):
#                 for j in range(6):
#                     print(r1[-1-c][i][j])
#                 print()
        pool1_conv_3 = np.zeros((128,64,64), dtype='float16')
        #卷积次数
        for c in range(2):
            for i in range(4):
                for j in range(4):
                    temp_pool = np.lib.pad(r1[c*64:(c+1)*64,i*32:32+i*32, j*32:32+j*32],((0, 0), (0, 2), (0, 2)), 'constant')
                    pool1_conv_3[c*64:(c+1)*64, i*16:(i+1)*16, j*16:(j+1)*16] = self.x.pool(temp_pool)
#         print('after max pooling -- pool1 shape : ' + str(pool1_conv_3.shape))
        
#         print('pool1')
#         for c in range(1):
#             for i in range(3):
#                 for j in range(3):
#                     print(pool1_conv_3[-1-c][i][j])
#                 print()
                
        #for i in range(8):
        #    print(pool1_conv_3[-1-i][i][i])
        '''
        状态：待验证
        实现redsidual 3*3 (pad+conv+relu)
        输入: pool1
        输出：r3
        输入尺寸: (128,64,64)
        输出尺寸: (128,64,64)
        卷积尺寸: (128,128,3,3)
        输入有效: (128,32,32)
        输出有效: (32,32,32)
        阶段: 3
        '''
        r3 = self._residual_64_64(pool1_conv_3,'r_3')
#         print('preprocess over!!!')

        return r3


    def _hourglass_stage_1(self, inputs):
#################################################  HG_Start  #################################################
        '''
        状态：待验证
        实现 redsidual 64*64 3*3 (pad+conv+relu)
        输入: inputs
        输出：up_1
        输入尺寸: (128,64,64)
        输出尺寸: (128,64,64)
        卷积尺寸: (128,128,3,3)
        输入有效: (128,32,32)
        输出有效: (32,32,32)
        阶段: 4
        '''
        up_1 = self._residual_64_64(inputs,'r_4')
#         print('up_1')
#         print(up_1.shape)

        '''
        状态：待验证
        实现 Maxpooling 2*2
        输入: inputs
        输出：low_
        输入尺寸: (128,64,64)
        输出尺寸: (128,32,32)
        池化尺寸: (2，2)
        输入有效: (128,32,32)
        输出有效: (32,16,16)
        '''

        low_ = np.zeros((128,32,32), dtype='float16')
            #卷积次数
        #b = inputs.swapaxes(0,1)
        #inputs_b = b.swapaxes(1,2)
        #inputs_b = np.expand_dims(inputs_b,axis = 0)
        #low_ = tf.contrib.layers.max_pool2d(inputs_b, [2,2], [2,2], padding='VALID')
        
        #low_ = low_.eval()
        #c = low_[0]
        #d = c.swapaxes(1,2)
        #low_ = d.swapaxes(0,1)
#         print(low_.shape)
        for c in range(2):
            for i in range(2):
                for j in range(2):
                    temp_low_ = np.lib.pad(inputs[c*64:(c+1)*64, i*32:32+i*32, j*32:32+j*32],((0, 0), (0, 2), (0, 2)), 'constant')
                    low_[c*64:(c+1)*64, i*16:(i+1)*16, j*16:(j+1)*16] = self.x.pool(temp_low_)
        
        
        
#         for i in range(2):
#             for j in range(2):
#                 print(low_[60][i][j])
#             print()
        
#         print('low_')
        #low_ = tf.contrib.layers.max_pool2d(inputs, [2,2], [2,2], padding='VALID')
        '''
        状态：待验证
        实现 redsidual 32*32 3*3 (pad+conv+relu)
        输入: low_
        输出：low_1
        输入尺寸: (128,32,32)
        输出尺寸: (128,32,32)
        卷积尺寸: (128,128,3,3)
        输入有效: (128,32,32)
        输出有效: (32,32,32)
        阶段: 5
        '''        
        low_1 = self._residual_32_32(low_,'r_5') 
#         print('low_1')
        '''
        状态：待验证
        实现 redsidual 32*32 3*3 (pad+conv+relu)
        输入: low_1
        输出：low_2_up_1
        输入尺寸: (128,32,32)
        输出尺寸: (128,32,32)
        卷积尺寸: (128,128,3,3)
        输入有效: (128,32,32)
        输出有效: (32,32,32)
        阶段: 6
        '''     
        low_2_up_1 = self._residual_32_32(low_1,'r_6') 
#         print('low_2_up_1')

        '''
        状态：待验证
        实现 Maxpooling 2*2
        输入: low_1
        输出：low_2_low_
        输入尺寸: (128,32,32)
        输出尺寸: (128,16,16)
        池化尺寸: (2，2)
        输入有效: (32,32,32)
        输出有效: (32,16,16)
        '''
        low_2_low_ = np.zeros((128,16,16), dtype='float16')
            #卷积次数
        for c in range(2):
            temp_low_2_up_1 = np.lib.pad(low_1[c*64:(c+1)*64],((0, 0), (0, 2), (0, 2)), 'constant')
            low_2_low_[c*64:(c+1)*64] = self.x.pool(temp_low_2_up_1)
#         print('low_2_low_')

        #low_ = tf.contrib.layers.max_pool2d(inputs, [2,2], [2,2], padding='VALID')


        '''
        状态：待验证
        实现 redsidual 16*16 3*3 (pad+conv+relu)
        输入: low_2_low_
        输出：low_2_low_1
        输入尺寸: (128,16,16)
        输出尺寸: (128,16,16)
        卷积尺寸: (128,128,3,3)
        输入有效: (128,16,16)
        输出有效: (32,16,16)
        阶段: 7
        ''' 
        low_2_low_1 = self._residual_16_16(low_2_low_,'r_7') 
#         print('low_2_low_1')

        '''
        状态：待验证
        实现 redsidual 16*16 3*3 (pad+conv+relu)
        输入: low_2_low_1
        输出：low_2_low_2_up_1
        输入尺寸: (128,16,16)
        输出尺寸: (128,16,16)
        卷积尺寸: (128,128,3,3)
        输入有效: (128,16,16)
        输出有效: (32,16,16)
        阶段: 8
        ''' 
        low_2_low_2_up_1 = self._residual_16_16(low_2_low_1,'r_8') 
#         print('low_2_low_2_up_1')

        '''
        状态：待验证
        实现 Maxpooling 2*2
        输入: low_2_low_1
        输出：low_2_low_2_low_
        输入尺寸: (128,16,16)
        输出尺寸: (128,8,8)
        池化尺寸: (2，2)
        输入有效: (32,16,16)
        输出有效: (32,8,8)
        '''
        low_2_low_2_low_ = np.zeros((128,8,8), dtype='float16')
            #卷积次数
        for c in range(2):
            temp_low_2_low_2_up_1 = np.lib.pad(low_2_low_1[c*64:(c+1)*64],((0, 0), (0, 18), (0, 18)), 'constant')
            temp_temp_low_2_low_2_up_1 = self.x.pool(temp_low_2_low_2_up_1)
            low_2_low_2_low_[c*64:(c+1)*64] = temp_temp_low_2_low_2_up_1[:,0:8,0:8]

        #low_ = tf.contrib.layers.max_pool2d(inputs, [2,2], [2,2], padding='VALID')
#         print('low_2_low_2_low_')


        '''
        状态：待验证
        实现 redsidual 8*8 3*3 (pad+conv+relu)
        输入: low_2_low_2_low_
        输出：low_2_low_2_low_1
        输入尺寸: (128,8,8)
        输出尺寸: (128,8,8)
        卷积尺寸: (128,128,3,3)
        输入有效: (128,8,8)
        输出有效: (32,8,8)
        阶段: 9
        ''' 
        low_2_low_2_low_1 = self._residual_8_8(low_2_low_2_low_,'r_9') 
#         print('')
        '''
        状态：待验证
        实现 redsidual 8*8 3*3 (pad+conv+relu)
        输入: low_2_low_2_low_1
        输出：low_2_low_2_low_2
        输入尺寸: (128,8,8)
        输出尺寸: (128,8,8)
        卷积尺寸: (128,128,3,3)
        输入有效: (128,8,8)
        输出有效: (32,8,8)
        阶段: 10
        ''' 
        low_2_low_2_low_2 = self._residual_8_8(low_2_low_2_low_1,'r_10') 
#         print('low_2_low_2_low_2')
        '''
        实现 redsidual 8*8 3*3 (pad+conv+relu)
        输入: low_2_low_2_low_2
        输出：low_2_low_2_low_3
        输入尺寸: (128,8,8)
        输出尺寸: (128,8,8)
        卷积尺寸: (128,128,3,3)
        输入有效: (128,8,8)
        输出有效: (32,8,8)
        阶段: 11
        ''' 
        low_2_low_2_low_3 = self._residual_8_8(low_2_low_2_low_2,'r_11') 
#         print('')
        '''
        实现 resize_nearest_neighbor
        输入: low_2_low_2_low_3
        输出：low_2_low_2_up_2
        输入尺寸: (128,8,8)
        输出尺寸: (128,16,16)
        采样尺寸: (2，2)
        输入有效: (32,8,8)
        输出有效: (32,16,16)
        '''
        #结果
        low_2_low_2_up_2 = np.zeros((128,16,16), dtype='float16')
        #先将(128,8,8)扩展到(128,16,16)
        
        for c in range(2):
            map_in_8_16 = np.lib.pad(low_2_low_2_low_3[c*64:(c+1)*64],((0, 0), (0, 26), (0, 26)), 'constant')
            temp_low_2_low_2_low_3 = self.x.up(map_in_8_16)
            low_2_low_2_up_2[c*64:(c+1)*64] = temp_low_2_low_2_low_3[:,0:16,0:16]

#         print('low_2_low_2_up_2')
        '''
        实现 HG_add
        输入: low_2_low_2_up_2
        输入: low_2_low_2_up_1
        输出：low_2_low_2
        输入尺寸: (128,16,16)
        输入尺寸: (128,16,16)
        输出尺寸: (128,16,16)
        ''' 
        low_2_low_2 = low_2_low_2_up_2 + low_2_low_2_up_1
#         print('low_2_low_2')
        '''
        实现 redsidual 16*16 3*3 (pad+conv+relu)
        输入: low_2_low_2
        输出: low_2_low_3
        输入尺寸: (128,16,16)
        输出尺寸: (128,16,16)
        卷积尺寸: (128,128,3,3)
        输入有效: (128,16,16)
        输出有效: (32,16,16)
        阶段: 12
        ''' 
        low_2_low_3 = self._residual_16_16(low_2_low_2,'r_12')
#         print('low_2_low_3')

        '''
        实现 resize_nearest_neighbor
        输入: low_2_low_3
        输出: low_2_up_2
        输入尺寸: (128,16,16)
        输出尺寸: (128,32,32)
        采样尺寸: (2，2)
        输入有效: (32,16,16)
        输出有效: (32,32,32)
        '''
        #结果
        low_2_up_2 = np.zeros((128,32,32), dtype='float16')
        for c in range(2):
            map_in_16_32 = np.lib.pad(low_2_low_3[c*64:(c+1)*64],((0, 0), (0, 18), (0, 18)), 'constant')
            low_2_up_2[c*64:(c+1)*64] = self.x.up(map_in_16_32)
#         print('low_2_up_2')
        '''
        实现 HG_add
        输入: low_2_up_2
        输入: low_2_up_1
        输出: low_2
        输入尺寸: (128,32,32)
        输入尺寸: (128,32,32)
        输出尺寸: (128,32,32)
        ''' 
        low_2 = low_2_up_1 + low_2_up_2
#         print('low_2')
        '''
        实现 redsidual 32*32 3*3 (pad+conv+relu)
        输入: low_2
        输出: low_3
        输入尺寸: (128,32,32)
        输出尺寸: (128,32,32)
        卷积尺寸: (128,128,3,3)
        输入有效: (128,32,32)
        输出有效: (32,32,32)
        阶段: 13
        ''' 
        low_3 = self._residual_32_32(low_2,'r_13')
#         print('low_3')
        '''
        实现 resize_nearest_neighbor
        输入: low_3
        输出: up_2
        输入尺寸: (128,32,32)
        输出尺寸: (128,64,64)
        采样尺寸: (2，2)
        输入有效: (32,16,16)
        输出有效: (32,32,32)
        '''
        up_2 = np.zeros((128,64,64), dtype='float16')
        for c in range(2):
            for i in range(2):
                for j in range(2):
                    map_in_32_64 = np.lib.pad(low_3[c*64:(c+1)*64, i*16:(i+1)*16, j*16:(j+1)*16],((0, 0), (0, 18), (0, 18)), 'constant')
                    up_2[c*64:(c+1)*64, i*32:(i+1)*32, j*32:(j+1)*32] = self.x.up(map_in_32_64)
#         print('up_2')
        '''
        实现 HG_add
        输入: up_2
        输入: up_1
        输出：temp
        输入尺寸: (128,64,64)
        输入尺寸: (128,64,64)
        输出尺寸: (128,64,64)
        ''' 
#         print('hourglass over!!!!')
        return up_1 + up_2
#################################################  HG_End  #################################################
    def _hourglass_stage_2(self, inputs):
#################################################  HG_Start  #################################################
        '''
        状态：待验证
        实现 redsidual 64*64 3*3 (pad+conv+relu)
        输入: inputs
        输出：up_1
        输入尺寸: (128,64,64)
        输出尺寸: (128,64,64)
        卷积尺寸: (128,128,3,3)
        输入有效: (128,32,32)
        输出有效: (32,32,32)
        阶段: 17
        '''
        up_1 = self._residual_64_64(inputs,'r_17')
#         print('up_1')
#         print(up_1.shape)

        '''
        状态：待验证
        实现 Maxpooling 2*2
        输入: inputs
        输出：low_
        输入尺寸: (128,64,64)
        输出尺寸: (128,32,32)
        池化尺寸: (2，2)
        输入有效: (128,32,32)
        输出有效: (32,16,16)
        '''

        low_ = np.zeros((128,32,32), dtype='float16')
            #卷积次数
        #b = inputs.swapaxes(0,1)
        #inputs_b = b.swapaxes(1,2)
        #inputs_b = np.expand_dims(inputs_b,axis = 0)
        #low_ = tf.contrib.layers.max_pool2d(inputs_b, [2,2], [2,2], padding='VALID')

        #low_ = low_.eval()
        #c = low_[0]
        #d = c.swapaxes(1,2)
        #low_ = d.swapaxes(0,1)
#         print(low_.shape)
        for c in range(2):
            for i in range(2):
                for j in range(2):
                    temp_low_ = np.lib.pad(inputs[c*64:(c+1)*64, i*32:32+i*32, j*32:32+j*32],((0, 0), (0, 2), (0, 2)), 'constant')
                    low_[c*64:(c+1)*64, i*16:(i+1)*16, j*16:(j+1)*16] = self.x.pool(temp_low_)



#         for i in range(2):
#             for j in range(2):
#                 print(low_[60][i][j])
#             print()

#         print('low_')
        #low_ = tf.contrib.layers.max_pool2d(inputs, [2,2], [2,2], padding='VALID')
        '''
        状态：待验证
        实现 redsidual 32*32 3*3 (pad+conv+relu)
        输入: low_
        输出：low_1
        输入尺寸: (128,32,32)
        输出尺寸: (128,32,32)
        卷积尺寸: (128,128,3,3)
        输入有效: (128,32,32)
        输出有效: (32,32,32)
        阶段: 18
        '''        
        low_1 = self._residual_32_32(low_,'r_18') 
#         print('low_1')
        '''
        状态：待验证
        实现 redsidual 32*32 3*3 (pad+conv+relu)
        输入: low_1
        输出：low_2_up_1
        输入尺寸: (128,32,32)
        输出尺寸: (128,32,32)
        卷积尺寸: (128,128,3,3)
        输入有效: (128,32,32)
        输出有效: (32,32,32)
        阶段: 19
        '''     
        low_2_up_1 = self._residual_32_32(low_1,'r_19') 
#         print('low_2_up_1')

        '''
        状态：待验证
        实现 Maxpooling 2*2
        输入: low_1
        输出：low_2_low_
        输入尺寸: (128,32,32)
        输出尺寸: (128,16,16)
        池化尺寸: (2，2)
        输入有效: (32,32,32)
        输出有效: (32,16,16)
        '''
        low_2_low_ = np.zeros((128,16,16), dtype='float16')
            #卷积次数
        for c in range(2):
            temp_low_2_up_1 = np.lib.pad(low_1[c*64:(c+1)*64],((0, 0), (0, 2), (0, 2)), 'constant')
            low_2_low_[c*64:(c+1)*64] = self.x.pool(temp_low_2_up_1)
#         print('low_2_low_')

        #low_ = tf.contrib.layers.max_pool2d(inputs, [2,2], [2,2], padding='VALID')


        '''
        状态：待验证
        实现 redsidual 16*16 3*3 (pad+conv+relu)
        输入: low_2_low_
        输出：low_2_low_1
        输入尺寸: (128,16,16)
        输出尺寸: (128,16,16)
        卷积尺寸: (128,128,3,3)
        输入有效: (128,16,16)
        输出有效: (32,16,16)
        阶段: 20
        ''' 
        low_2_low_1 = self._residual_16_16(low_2_low_,'r_20') 
#         print('low_2_low_1')

        '''
        状态：待验证
        实现 redsidual 16*16 3*3 (pad+conv+relu)
        输入: low_2_low_1
        输出：low_2_low_2_up_1
        输入尺寸: (128,16,16)
        输出尺寸: (128,16,16)
        卷积尺寸: (128,128,3,3)
        输入有效: (128,16,16)
        输出有效: (32,16,16)
        阶段: 21
        ''' 
        low_2_low_2_up_1 = self._residual_16_16(low_2_low_1,'r_21') 
#         print('low_2_low_2_up_1')

        '''
        状态：待验证
        实现 Maxpooling 2*2
        输入: low_2_low_1
        输出：low_2_low_2_low_
        输入尺寸: (128,16,16)
        输出尺寸: (128,8,8)
        池化尺寸: (2，2)
        输入有效: (32,16,16)
        输出有效: (32,8,8)
        '''
        low_2_low_2_low_ = np.zeros((128,8,8), dtype='float16')
            #卷积次数
        for c in range(2):
            temp_low_2_low_2_up_1 = np.lib.pad(low_2_low_1[c*64:(c+1)*64],((0, 0), (0, 18), (0, 18)), 'constant')
            temp_temp_low_2_low_2_up_1 = self.x.pool(temp_low_2_low_2_up_1)
            low_2_low_2_low_[c*64:(c+1)*64] = temp_temp_low_2_low_2_up_1[:,0:8,0:8]

        #low_ = tf.contrib.layers.max_pool2d(inputs, [2,2], [2,2], padding='VALID')
#         print('low_2_low_2_low_')


        '''
        状态：待验证
        实现 redsidual 8*8 3*3 (pad+conv+relu)
        输入: low_2_low_2_low_
        输出：low_2_low_2_low_1
        输入尺寸: (128,8,8)
        输出尺寸: (128,8,8)
        卷积尺寸: (128,128,3,3)
        输入有效: (128,8,8)
        输出有效: (32,8,8)
        阶段: 22
        ''' 
        low_2_low_2_low_1 = self._residual_8_8(low_2_low_2_low_,'r_22') 
#         print('')
        '''
        状态：待验证
        实现 redsidual 8*8 3*3 (pad+conv+relu)
        输入: low_2_low_2_low_1
        输出：low_2_low_2_low_2
        输入尺寸: (128,8,8)
        输出尺寸: (128,8,8)
        卷积尺寸: (128,128,3,3)
        输入有效: (128,8,8)
        输出有效: (32,8,8)
        阶段: 23
        ''' 
        low_2_low_2_low_2 = self._residual_8_8(low_2_low_2_low_1,'r_23') 
#         print('low_2_low_2_low_2')
        '''
        实现 redsidual 8*8 3*3 (pad+conv+relu)
        输入: low_2_low_2_low_2
        输出：low_2_low_2_low_3
        输入尺寸: (128,8,8)
        输出尺寸: (128,8,8)
        卷积尺寸: (128,128,3,3)
        输入有效: (128,8,8)
        输出有效: (32,8,8)
        阶段: 24
        ''' 
        low_2_low_2_low_3 = self._residual_8_8(low_2_low_2_low_2,'r_24') 
#         print('')
        '''
        实现 resize_nearest_neighbor
        输入: low_2_low_2_low_3
        输出：low_2_low_2_up_2
        输入尺寸: (128,8,8)
        输出尺寸: (128,16,16)
        采样尺寸: (2，2)
        输入有效: (32,8,8)
        输出有效: (32,16,16)
        '''
        #结果
        low_2_low_2_up_2 = np.zeros((128,16,16), dtype='float16')
        #先将(128,8,8)扩展到(128,16,16)

        for c in range(2):
            map_in_8_16 = np.lib.pad(low_2_low_2_low_3[c*64:(c+1)*64],((0, 0), (0, 26), (0, 26)), 'constant')
            temp_low_2_low_2_low_3 = self.x.up(map_in_8_16)
            low_2_low_2_up_2[c*64:(c+1)*64] = temp_low_2_low_2_low_3[:,0:16,0:16]

#         print('low_2_low_2_up_2')
        '''
        实现 HG_add
        输入: low_2_low_2_up_2
        输入: low_2_low_2_up_1
        输出：low_2_low_2
        输入尺寸: (128,16,16)
        输入尺寸: (128,16,16)
        输出尺寸: (128,16,16)
        ''' 
        low_2_low_2 = low_2_low_2_up_2 + low_2_low_2_up_1
#         print('low_2_low_2')
        '''
        实现 redsidual 16*16 3*3 (pad+conv+relu)
        输入: low_2_low_2
        输出: low_2_low_3
        输入尺寸: (128,16,16)
        输出尺寸: (128,16,16)
        卷积尺寸: (128,128,3,3)
        输入有效: (128,16,16)
        输出有效: (32,16,16)
        阶段: 25
        ''' 
        low_2_low_3 = self._residual_16_16(low_2_low_2,'r_25')
#         print('low_2_low_3')

        '''
        实现 resize_nearest_neighbor
        输入: low_2_low_3
        输出: low_2_up_2
        输入尺寸: (128,16,16)
        输出尺寸: (128,32,32)
        采样尺寸: (2，2)
        输入有效: (32,16,16)
        输出有效: (32,32,32)
        '''
        #结果
        low_2_up_2 = np.zeros((128,32,32), dtype='float16')
        for c in range(2):
            map_in_16_32 = np.lib.pad(low_2_low_3[c*64:(c+1)*64],((0, 0), (0, 18), (0, 18)), 'constant')
            low_2_up_2[c*64:(c+1)*64] = self.x.up(map_in_16_32)
#         print('low_2_up_2')
        '''
        实现 HG_add
        输入: low_2_up_2
        输入: low_2_up_1
        输出: low_2
        输入尺寸: (128,32,32)
        输入尺寸: (128,32,32)
        输出尺寸: (128,32,32)
        ''' 
        low_2 = low_2_up_1 + low_2_up_2
#         print('low_2')
        '''
        实现 redsidual 32*32 3*3 (pad+conv+relu)
        输入: low_2
        输出: low_3
        输入尺寸: (128,32,32)
        输出尺寸: (128,32,32)
        卷积尺寸: (128,128,3,3)
        输入有效: (128,32,32)
        输出有效: (32,32,32)
        阶段: 26
        ''' 
        low_3 = self._residual_32_32(low_2,'r_26')
#         print('low_3')
        '''
        实现 resize_nearest_neighbor
        输入: low_3
        输出: up_2
        输入尺寸: (128,32,32)
        输出尺寸: (128,64,64)
        采样尺寸: (2，2)
        输入有效: (32,16,16)
        输出有效: (32,32,32)
        '''
        up_2 = np.zeros((128,64,64), dtype='float16')
        for c in range(2):
            for i in range(2):
                for j in range(2):
                    map_in_32_64 = np.lib.pad(low_3[c*64:(c+1)*64, i*16:(i+1)*16, j*16:(j+1)*16],((0, 0), (0, 18), (0, 18)), 'constant')
                    up_2[c*64:(c+1)*64, i*32:(i+1)*32, j*32:(j+1)*32] = self.x.up(map_in_32_64)
#         print('up_2')
        '''
        实现 HG_add
        输入: up_2
        输入: up_1
        输出：temp
        输入尺寸: (128,64,64)
        输入尺寸: (128,64,64)
        输出尺寸: (128,64,64)
        ''' 
#         print('hourglass over!!!!')
        return up_1 + up_2

    def _predict(self, hg):
        j = np.ones(shape = (16,2)) * -1
        for i in range(len(j)):
                idx = np.unravel_index( hg[0,:,:,i].argmax(), (64,64))	#得到第i个节点中概率最大的那个点在图像中的索引值（在像素为64*64图像中的位置）
                print(idx)
                print(hg[0, idx[0], idx[1], i])
                if hg[0, idx[0], idx[1], i] > self.thresh:	#如果该点的概率大于阈值
                    j[i] = np.asarray(idx) * 800 / 64	#得到i节点在原图中的坐标
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
#         print('over!!!!!!!!!!!!!!!!!!!!!!')
        cv2.imshow('stream', self.img_res)
        cv2.waitKey(0)
    
    def _conv_128_64_64(self,inputs):
        '''
        状态：√
        实现 conv_bn_relu
        输入: hg[1]
        输出：ll[1]
        输入尺寸: (128,64,64)
        输出尺寸: (128,64,64)
        卷积尺寸: (128,128,1,1)
        输入有效: (128,32,32)
        输出有效: (32,32,32)
        阶段: 27
        ''' 
        #准备数据
        ll = np.zeros((128,64,64), dtype='float16')
        hg_1_weights = self.W['r_27'][0]
        hg_1_beta = self.W['r_27'][1]
        for c in range(2): 
                #拆分
                for i in range(2): 
                    for j in range(2):
                        #卷积 each valid shape:(32,32,32)
                        map_in_ll_1_temp = np.lib.pad(inputs[:, i*32:32+i*32, j*32:32+j*32],((0, 0), (0, 2), (0, 2)), 'constant')
                        ll[c*64:(c+1)*64, i*32:(i+1)*32, j*32:(j+1)*32] = self.x.conv_32_32(map_in_ll_1_temp, hg_1_weights[:,c*64:(c+1)*64,:,:], hg_1_beta[c*64:(c+1)*64])
        return ll
    
    def _conv_128_16(self, inputs):
        '''
        状态：√
        实现 conv2d (no pad + no bn + no relu)
        输入: ll[1]
        输出：out[1]
        输入尺寸: (128,64,64)
        输出尺寸: (16,64,64)
        卷积尺寸: (128,16,1,1)
        输入有效: (128,32,32)
        输出有效: (16,32,32)
        阶段: 28
        '''
        out = np.zeros((16,64,64), dtype='float16')
        ll_1_weights = self.W['r_28'][0]
        print(ll_1_weights.shape)
        ll_1_weights = np.lib.pad(ll_1_weights,((0, 0), (0, 32), (0, 0), (0, 0)), 'constant')
        ll_1_beta = np.zeros((64), dtype='float16')
        for i in range(2): 
            for j in range(2):
                #卷积 each valid shape:(32,32,32)
                map_in_temp_ll_1 = np.lib.pad(inputs[:, i*32:32+i*32, j*32:32+j*32],((0, 0), (0, 2), (0, 2)), 'constant')
                
                out[:, i*32:(i+1)*32, j*32:(j+1)*32] = self.x.conv_out_16(map_in_temp_ll_1, ll_1_weights, ll_1_beta)
        return out
    def _sigmoid(self, inputs):
        '''
        状态：待验证
        实现 sigmoid
        输入: out[1]
        输出：hgg
        输入尺寸: (16,64,64)
        输出尺寸: (16,64,64)
        输入有效: (16,32,32)
        输出有效: (16,32,32)
        阶段: 29
        ''' 
        hgg = np.zeros((16,64,64),dtype='float16')
        for i in range(2): 
            for j in range(2):
                #卷积 each valid shape:(32,32,32)
                map_in_out_temp = np.lib.pad(inputs[:, i*32:32+i*32, j*32:32+j*32],((0, 48), (0, 2), (0, 2)), 'constant')
                hgg[:, i*32:(i+1)*32, j*32:(j+1)*32] = self.x.sigmoid(map_in_out_temp)
                
        return hgg  
    
    def HG(self):
        #开始读取图片
        print('start!')
        all_start = t.time()
        self._get_img()
        
        #开始读取视频
        #self._get_video()
        #预处理
        tesmp_1 = t.time()
        
        _hourglass_input = self._preprocessing()

        hg = [None] * self.nStack
        ll = [None] * self.nStack
        #ll_ = [None] * self.nStack
        #drop = [None] * self.nStack
        out = [None] * self.nStack
        out_ = [None] * self.nStack
        sum_ = [None] * self.nStack
        tesmp_2 = t.time()
        print('Preprocessing_end :')
        print(tesmp_2-tesmp_1)
        hg[0] = self._hourglass_stage_1(_hourglass_input)
        
        '''
        状态：√
        实现 conv_bn_relu
        输入: hg[0]
        输出：ll[0]
        输入尺寸: (128,64,64)
        输出尺寸: (128,64,64)
        卷积尺寸: (128,128,1,1)
        输入有效: (128,32,32)
        输出有效: (32,32,32)
        阶段: 14
        '''  
        #准备数据
        ll[0] = np.zeros((128,64,64), dtype='float16')
        hg_0_weights = self.W['r_14'][0]
        hg_0_beta = self.W['r_14'][1]
        for c in range(2): 
                #拆分
                for i in range(2): 
                    for j in range(2):
                        #卷积 each valid shape:(32,32,32)
                        map_in_temp_hg_0 = np.lib.pad(hg[0][:, i*32:32+i*32, j*32:32+j*32],((0, 0), (0, 2), (0, 2)), 'constant')
                        ll[0][c*64:(c+1)*64, i*32:(i+1)*32, j*32:(j+1)*32] = self.x.conv_32_32(map_in_temp_hg_0, hg_0_weights[:,c*64:(c+1)*64,:,:], hg_0_beta[c*64:(c+1)*64])
#         print('ll[0]')
#         for i in range(8):
#             print(ll[0][-1-i][i][i])

        '''
        状态：√
        实现 conv2d (no pad + no bn + no relu)
        输入: ll[0]
        输出：out[0]
        输入尺寸: (128,64,64)
        输出尺寸: (16,64,64)
        卷积尺寸: (128,16,1,1)
        输入有效: (128,32,32)
        输出有效: (16,32,32)
        阶段: 15
        ''' 
        #准备数据
        out[0] = np.zeros((16,64,64), dtype='float16')
        ll_0_weights = self.W['r_15'][0]
        ll_0_beta = np.zeros((64), dtype='float16')
        ll_0_weights = np.lib.pad(ll_0_weights,((0, 0), (0, 32), (0, 0), (0, 0)), 'constant')
        for i in range(2): 
            for j in range(2):
                #卷积 each valid shape:(32,32,32)
                map_in_temp_ll_0 = np.lib.pad(ll[0][:, i*32:32+i*32, j*32:32+j*32],((0, 0), (0, 2), (0, 2)), 'constant')
                out[0][:, i*32:(i+1)*32, j*32:(j+1)*32] = self.x.conv_out_16(map_in_temp_ll_0, ll_0_weights, ll_0_beta)
#         print('out[0]')
#         for i in range(8):
#             print(out[0][-1-i][i][i])
        '''
        状态：√
        实现 conv2d (no pad + no bn + no relu)
        输入: out[0]
        输出：out_[0]
        输入尺寸: (16,64,64)
        输出尺寸: (128,64,64)
        卷积尺寸: (16,128,1,1)
        输入有效: (16,32,32)
        输出有效: (32,32,32)
        阶段: 16
        ''' 
        out_[0] = np.zeros((128,64,64), dtype='float16')
        out_0_weights = self.W['r_16'][0]
        out_0_beta = np.zeros((64), dtype='float16')
        #将输入从16层变为128层
        map_in_out0 = np.lib.pad(out[0],((0, 112), (0, 0), (0, 0)), 'constant')
        for c in range(2):
            for i in range(2): 
                for j in range(2):
                    #卷积 each valid shape:(32,32,32)
                    
                    map_in_out0_temp = np.lib.pad(map_in_out0[:, i*32:32+i*32, j*32:32+j*32],((0, 0), (0, 2), (0, 2)), 'constant')
                    out_[0][c*64:(c+1)*64, i*32:(i+1)*32, j*32:(j+1)*32] = self.x.conv_32_32_without_relu(map_in_out0_temp, out_0_weights[:,c*64:(c+1)*64,:,:], out_0_beta)
#         print('out_[0]')
#         for i in range(8):
#             print(out_[0][-1-i][i][i])
        '''
        实现 HG_add
        输入: out_[0]
        输入: ll[0]
        输出：sum_[0]
        输入尺寸: (128,64,64)
        输入尺寸: (128,64,64)
        输出尺寸: (128,64,64)
        ''' 
        sum_[0] = out_[0] + ll[0] + _hourglass_input
#         print('sum_[0]')

        hg[1] = self._hourglass_stage_2(sum_[0])
        
        '''
        状态：√
        实现 conv_bn_relu
        输入: hg[1]
        输出：ll[1]
        输入尺寸: (128,64,64)
        输出尺寸: (128,64,64)
        卷积尺寸: (128,128,1,1)
        输入有效: (128,32,32)
        输出有效: (32,32,32)
        阶段: 27
        ''' 
        ll[1] = self._conv_128_64_64(hg[1])
#         print('ll[1]')
#         for i in range(8):
#             print(ll[1][-1-i][i][i])
        '''
        状态：√
        实现 conv2d (no pad + no bn + no relu)
        输入: ll[1]
        输出：out[1]
        输入尺寸: (128,64,64)
        输出尺寸: (16,64,64)
        卷积尺寸: (128,16,1,1)
        输入有效: (128,32,32)
        输出有效: (16,32,32)
        阶段: 28
        ''' 
        
#         print('out[1]')
        #准备数据
        out[1] = self._conv_128_16(ll[1])
#         for i in range(8):
#             print(out[1][-1-i][i][i])
#         print('almost there !!!!!')
        '''
        状态：待验证
        实现 sigmoid
        输入: out[1]
        输出：hgg
        输入尺寸: (16,64,64)
        输出尺寸: (16,64,64)
        输入有效: (16,32,32)
        输出有效: (16,32,32)
        阶段: 29
        ''' 
        hgg = self._sigmoid(out[1])
        tesmp_3 = t.time()
        print('HG_end :')
        print(tesmp_3-tesmp_2)
        print('Model_end :')
        print(tesmp_3-all_start)
        print()
        
                #hgg[:, i*32:(i+1)*32, j*32:(j+1)*32] = self.x.sigmoid(map_in_out_temp)
#         print('hgg')
#         print(hgg.shape)
        return hgg
    def predict(self, inputs):
#         for i in range(8):
#             print(inputs[i][i][i])
        hgg = inputs.swapaxes(0,1)
        hgg = hgg.swapaxes(1,2)
        hgg = np.expand_dims(hgg,axis = 0)
        
        self._predict(hgg)
        
    def video_detect(self):
        cam = cv2.VideoCapture(0)
        while True:
        #for i in range(1):
            #读取视频
            Video_1 = t.time()
            ret_val, img = cam.read()
            img = cv2.flip(img, 1)	#图像水平翻转
            img[:, self.cam_res[1]//2 - self.cam_res[0]//2:self.cam_res[1]//2 + self.cam_res[0]//2]	#切片是左闭右开 ，高不变，宽变为(80:560)
            self.img_res = cv2.resize(img, (256,256))
            img_hg = cv2.resize(img, (256,256))
            img_hg = cv2.cvtColor(img_hg, cv2.COLOR_BGR2RGB)
            img_hg = img_hg.astype(np.float16)
            #调换维度(256,256,3) -> (3，256，256)
            temp = img_hg.swapaxes(1,2)
            img_hg = temp.swapaxes(0,1)
            self.img_hg = img_hg/255
            self.img_hg = self.img_hg.astype(np.float16)
            Video_2 = t.time()
            print('get img :')
            print(Video_2 - Video_1)
#             print(self.img_hg.dtype)
#             print('input image shape : ' + str(img_hg.shape))
            #分析图片
            
            hgg = self.HG()
            Video_3 = t.time()
#             print('HourGlass 1+2 :')
#             print(Video_3 - Video_2)
            #预测图片
            hg = hgg.swapaxes(0,1)
            hg = hg.swapaxes(1,2)
            hg = np.expand_dims(hg,axis = 0)
            j = np.ones(shape = (16,2)) * -1
            for i in range(len(j)):
                idx = np.unravel_index( hg[0,:,:,i].argmax(), (64,64))	#得到第i个节点中概率最大的那个点在图像中的索引值（在像素为64*64图像中的位置）
                print(idx)
                print(hg[0, idx[0], idx[1], i])
                if hg[0, idx[0], idx[1], i] > self.thresh:	#如果该点的概率大于阈值
                    j[i] = np.asarray(idx) * 256 / 64	#得到i节点在原图中的坐标
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
            Video_4 = t.time()
            print('predict :')
            print(Video_4 - Video_3)
            print('all time:')
            print(Video_4 - Video_1)
            cv2.imshow('stream', self.img_res)
            if cv2.waitKey(1) == 27:	#按下ESC键退出
                print('Stream Ended')
                cv2.destroyAllWindows()
                break
    
    def video_test(self):
        cam = cv2.VideoCapture(0)
        while True:
            ret_val, img = cam.read()
            img_hg = cv2.resize(img, (256,256))
            cv2.imshow('stream', img_hg)
            if cv2.waitKey(1) == 27:	#按下ESC键退出
                print('Stream Ended')
                cv2.destroyAllWindows()
                break
        
if __name__ == "__main__":
    test_final = HG_Xilinx()
    a = test_final.HG()
    test_final.predict(a)