# 函数名               功能                    尺寸
# conv_8_8  (weight全部bias全部）            8 x 8卷积                map_in --> 64x34x34 (64x10x10) \ weight --> 64x64x3x3 \ bias--> 64 \ out --> 64x8x8
# conv_32_32 (weight全部bias全部）           32 x32卷积                map_in --> 64x34x34 \ weight --> 64x64x3x3 \ bias--> 64 \ out --> 64x32x32
# conv_68_68 (weight全部bias全部）            68 x 68卷积               map_in --> 3x68x68 \ weight --> 3x64x6x6 \ bias--> 64 \ out --> 64x32x32
# pool                32x32 --> 16x16池化          map_in --> 64x34x34 (64x32x32) \ out --> 64x16x16
# up                 16x16 --> 32x32上采样         map_in --> 64x34x34 (64x16x16) \ out --> 64x32x32
# conv_32_32_without_relu (weight全部bias全部）  不加Relu的32 x 32卷积         map_in --> 64x34x34 \ weight --> 64x64x3x3 \ bias--> 64 \ out --> 64x32x32
# sigmoid              sigmoid                  map_in --> 64x34x34 (64x32x32) \ out --> 64x32x32
# conv_out_16 (weight全部bias全部）          输出深度为16的32 x 32卷积      map_in --> 64x34x34 \ weight --> 64x64x3x3(64x16x3x3) \ bias--> 64(16) \ out --> 16x32x32
# pool_16_8  (weight全部bias全部）          16x16 --> 32x32池化           map_in --> 64x34x34(64x16x16) \ out --> 64x8x8

import numpy as np
from pynq import Xlnk
from pynq import Overlay
import time

class Conv():
    def __init__(self):
        self.ol = Overlay("/home/xilinx/jupyter_notebooks/IP core/end.bit")
        self.dma_0 = self.ol.axi_dma_0
        self.dma_1 = self.ol.axi_dma_1
        self.dma_2 = self.ol.axi_dma_2
        self.dma_3 = self.ol.axi_dma_3
        self.dma_4 = self.ol.axi_dma_4
        self.dma_5 = self.ol.axi_dma_5
        self.top = self.ol.top_0
        
        xlnk = Xlnk()
        self.map_in_buffer = xlnk.cma_array(shape=(128,34,34), dtype=np.float16)
        self.weight_buffer = xlnk.cma_array(shape=(128,64,3,3), dtype=np.float16)
        self.bias_buffer = xlnk.cma_array(shape=(64), dtype=np.float16)
        self.out_buffer = xlnk.cma_array(shape=(64,32,32), dtype=np.float16)
        self.map_in_buffer_2 = xlnk.cma_array(shape=(3,68,68), dtype=np.float16)
        self.weight_buffer_2 = xlnk.cma_array(shape=(3,64,6,6), dtype=np.float16)
        self.bias_buffer_2 = xlnk.cma_array(shape=(64), dtype=np.float16)
        self.out_buffer_2 = xlnk.cma_array(shape=(64,32,32), dtype=np.float16)
        
        self.map_in_64 = xlnk.cma_array(shape=(64,34,34), dtype=np.float16)
        
    
        #map_in --> 64x34x34 (128x10x10)  weight --> 64x64x3x3  bias--> 64  out --> 64x8x8
    def conv_8_8(self,map_in,weight,bias):
        a = np.zeros((128,34,34),dtype='float16')
        a[:] = map_in
        self.map_in_buffer[:] = a
        self.weight_buffer[:] = weight
        self.bias_buffer[:] = bias
        
        self.top.write(0x10,0x01)
        self.dma_2.sendchannel.transfer(self.bias_buffer)
        self.dma_1.sendchannel.transfer(self.weight_buffer)
        self.dma_0.sendchannel.transfer(self.map_in_buffer)
        self.dma_0.recvchannel.transfer(self.out_buffer)
        self.top.write(0x00,0x01)
        self.dma_2.sendchannel.wait()
        self.dma_1.sendchannel.wait()
        self.dma_0.sendchannel.wait()
        self.dma_0.recvchannel.wait()
        
        return np.array(self.out_buffer[:,:8,:8])
    
    
        #map_in --> 64x34x34   weight --> 64x64x3x3  bias--> 64  out --> 64x32x32
    def conv_32_32(self,map_in,weight,bias):
        a = np.zeros((128,34,34),dtype='float16')
        a[:] = map_in
        self.map_in_buffer[:] = a
        #print(2)
        self.weight_buffer[:] = weight
        #print(3)
        self.bias_buffer[:] = bias
        #print(4)
        self.top.write(0x10,0x02)
        self.dma_2.sendchannel.transfer(self.bias_buffer)
        self.dma_1.sendchannel.transfer(self.weight_buffer)
        self.dma_0.sendchannel.transfer(self.map_in_buffer)
        self.dma_0.recvchannel.transfer(self.out_buffer)
        self.top.write(0x00,0x01)
        self.dma_2.sendchannel.wait()
        self.dma_1.sendchannel.wait()
        self.dma_0.sendchannel.wait()
        self.dma_0.recvchannel.wait()
        
        return np.array(self.out_buffer)
    
    
        #map_in --> 3x68x68   weight --> 3x64x6x6  bias--> 64  out --> 64x32x32
    def conv_68_68(self,map_in,weight,bias):
        a = np.zeros((3,68,68),dtype='float16')
        a[:] = map_in
        self.map_in_buffer_2[:] = a
        self.weight_buffer_2[:] = weight
        self.bias_buffer_2[:] = bias
        
        self.top.write(0x10,0x03)
        self.dma_3.sendchannel.transfer(self.map_in_buffer_2)
        self.dma_4.sendchannel.transfer(self.weight_buffer_2)
        self.dma_5.sendchannel.transfer(self.bias_buffer_2)
        self.dma_3.recvchannel.transfer(self.out_buffer_2)
        self.top.write(0x00,0x01)
        self.dma_5.sendchannel.wait()
        self.dma_4.sendchannel.wait()
        self.dma_3.sendchannel.wait()
        self.dma_3.recvchannel.wait()
        
        return np.array(self.out_buffer_2)
    
    
        #map_in --> 64x34x34 (64x32x32)  out --> 64x16x16
    def pool(self,map_in):
        a = np.zeros((64,34,34),dtype='float16')
        a[:] = map_in
        self.map_in_64[:] = a
        
        self.top.write(0x10,0x04)
        self.dma_0.sendchannel.transfer(self.map_in_64)
        self.dma_0.recvchannel.transfer(self.out_buffer)
        self.top.write(0x00,0x01)
        self.dma_0.sendchannel.wait()
        self.dma_0.recvchannel.wait()
        
        return np.array(self.out_buffer[:,:16,:16])
    
    
        #map_in --> 64x34x34 (64x16x16)  out --> 64x32x32
    def up(self,map_in):
        a = np.zeros((64,34,34),dtype='float16')
        a[:] = map_in
        self.map_in_64[:] = a
        
        self.top.write(0x10,0x05)
        self.dma_0.sendchannel.transfer(self.map_in_64)
        self.dma_0.recvchannel.transfer(self.out_buffer)
        self.top.write(0x00,0x01)
        self.dma_0.sendchannel.wait()
        self.dma_0.recvchannel.wait()
        
        return np.array(self.out_buffer)
    
    
        #map_in --> 64x34x34   weight --> 64x64x3x3  bias--> 64  out --> 64x32x32
    def conv_32_32_without_relu(self,map_in,weight,bias):
        a = np.zeros((128,34,34),dtype='float16')
        a[:] = map_in
        self.map_in_buffer[:] = a
        self.weight_buffer[:] = weight
        self.bias_buffer[:] = bias
        
        self.top.write(0x10,0x06)
        self.dma_2.sendchannel.transfer(self.bias_buffer)
        self.dma_1.sendchannel.transfer(self.weight_buffer)
        self.dma_0.sendchannel.transfer(self.map_in_buffer)
        self.dma_0.recvchannel.transfer(self.out_buffer)
        self.top.write(0x00,0x01)
        self.dma_2.sendchannel.wait()
        self.dma_1.sendchannel.wait()
        self.dma_0.sendchannel.wait()
        self.dma_0.recvchannel.wait()
        
        return np.array(self.out_buffer)
    
    
        #map_in --> 64x34x34 (64x32x32)  out --> 64x32x32
    def sigmoid(self,map_in):
        a = np.zeros((64,34,34),dtype='float16')
        a[:] = map_in
        self.map_in_64[:] = a
        
        self.top.write(0x10,0x07)
        self.dma_0.sendchannel.transfer(self.map_in_64)
        self.dma_0.recvchannel.transfer(self.out_buffer)
        self.top.write(0x00,0x01)
        self.dma_0.sendchannel.wait()
        self.dma_0.recvchannel.wait()
        
        return np.array(self.out_buffer[:16,:,:])
    
    
    #map_in --> 64x34x34   weight --> 64x32x3x3  bias--> 64  out --> 16x32x32
    def conv_out_16(self,map_in,weight,bias):
        a = np.zeros((128,34,34),dtype='float16')
        a[:] = map_in
        self.map_in_buffer[:] = a
        self.weight_buffer[:] = weight
        self.bias_buffer[:] = bias
        
        self.top.write(0x10,0x06)
        self.dma_2.sendchannel.transfer(self.bias_buffer)
        self.dma_1.sendchannel.transfer(self.weight_buffer)
        self.dma_0.sendchannel.transfer(self.map_in_buffer)
        self.dma_0.recvchannel.transfer(self.out_buffer)
        self.top.write(0x00,0x01)
        self.dma_2.sendchannel.wait()
        self.dma_1.sendchannel.wait()
        self.dma_0.sendchannel.wait()
        self.dma_0.recvchannel.wait()
        
        return np.array(self.out_buffer[:16,:,:])
    
    
    #map_in --> 64x16x16  out --> 64x8x8
    def pool_16_8(self,map_in):
        a = np.zeros((64,34,34),dtype='float16')
        a[:] = map_in
        self.map_in_64[:] = a
        
        self.top.write(0x10,0x04)
        self.dma_0.sendchannel.transfer(self.map_in_64)
        self.dma_0.recvchannel.transfer(self.out_buffer)
        self.top.write(0x00,0x01)
        self.dma_0.sendchannel.wait()
        self.dma_0.recvchannel.wait()
        
        return np.array(self.out_buffer[:,:8,:8])
    
'''
test = Conv()
map_in = np.zeros((128,34,34), dtype='float16')
weights = np.zeros((32,3,3), dtype='float16')
bias = np.zeros((32), dtype='float16')
test.conv_32_32(map_in,weights,bias)
'''  