# -*- coding: utf-8 -*-

import numpy


data = [1,2,3,4,5,6]
x = numpy.array(data)
print (x) #打印数组
print x.dtype #打印数组元素的类型


data = [[1,2],[3,4],[5,6]]
x = numpy.array(data)
print x #打印数组
print x.ndim #打印数组的维度
print x.shape #打印数组各个维度的长度。shape是一个元组


x = numpy.zeros(6) #创建一维长度为6的，元素都是0一维数组
print x
x = numpy.zeros((2,3)) #创建一维长度为2，二维长度为3的二维0数组
print x
x = numpy.ones((2,3)) #创建一维长度为2，二维长度为3的二维1数组
print x
x = numpy.empty((3,3)) #创建一维长度为2，二维长度为3,未初始化的二维数组
print x


print numpy.arange(6) # [0,1,2,3,4,5,] 开区间
print numpy.arange(0,6,2)  # [0, 2，4]
