#coding=utf-8
import caffe
import numpy as np
import matplotlib.pyplot as plt
import os
import PIL
from PIL import Image
import sys

#定义Caffe根目录
caffe_root = '/home/itachi/caffe/'
#网络结构描述文件
deploy_file = caffe_root+'icvl/deploy.prototxt'
#训练好的模型
model_file = caffe_root+'icvl/snapshot_iter_2656.caffemodel'

#cpu模式
# caffe.set_mode_cpu()
caffe.set_mode_gpu()
#定义网络模型
net = caffe.Classifier(deploy_file, #调用deploy文件
                       model_file,  #调用模型文件
                       channel_swap=(2,1,0),  #caffe中图片是BGR格式，而原始格式是RGB，所以要转化
                       raw_scale=255,         #python中将图片存储为[0, 1]，而caffe中将图片存储为[0, 255]，所以需要一个转换
                       image_dims=(224, 224),
                       mean=np.load(caffe_root +'icvl/mean.npy').mean(1).mean(1))#调用均值文件) #输入模型的图片要是224*224的图片

#分类标签文件
imagenet_labels_filename = caffe_root +'icvl/labels.txt'
#载入分类标签文件
labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')

#对目标路径中的图像，遍历并分类
def asd():
    for root,dirs,files in os.walk('/mnt/5858379E583779B8/VSC/Python/Machine Learning/icvl/whiteres2/5/'):
        for file in files:
            #加载要分类的图片
            image_file = os.path.join(root,file)
            input_image = caffe.io.load_image(image_file)

            #打印图片路径及名称
            image_path = os.path.join(root,file)
            print(image_path)
            
            #显示图片
            img=Image.open(image_path)
            plt.imshow(img)
            plt.axis('off')
            plt.show()
            
            #预测图片类别
            prediction = net.predict([input_image])
            print 'predicted class:',prediction[0].argmax()

            # 输出概率最大的前5个预测结果
            top_k = prediction[0].argsort()[-5:][::-1]
            for node_id in top_k:     
                #获取分类名称
                human_string = labels[node_id]
                #获取该分类的置信度
                score = prediction[0][node_id]
                print('%s (score = %.5f)' % (human_string, score))
if __name__ == "__main__":
    input_image = caffe.io.load_image('/mnt/5858379E583779B8/VSC/Python/Machine Learning/icvl/pose-ren/src/imgs/201912181395223520/gujia.png')
    pre = net.predict([input_image])
    print 'predicted class:',pre[0].argmax()
    asd()