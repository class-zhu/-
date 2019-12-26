#coding=utf-8
import caffe
import numpy as np
import matplotlib.pyplot as plt
import os
import PIL
from PIL import Image
import sys

#����Caffe��Ŀ¼
caffe_root = '/home/itachi/caffe/'
#����ṹ�����ļ�
deploy_file = caffe_root+'icvl/deploy.prototxt'
#ѵ���õ�ģ��
model_file = caffe_root+'icvl/snapshot_iter_2656.caffemodel'

#cpuģʽ
# caffe.set_mode_cpu()
caffe.set_mode_gpu()
#��������ģ��
net = caffe.Classifier(deploy_file, #����deploy�ļ�
                       model_file,  #����ģ���ļ�
                       channel_swap=(2,1,0),  #caffe��ͼƬ��BGR��ʽ����ԭʼ��ʽ��RGB������Ҫת��
                       raw_scale=255,         #python�н�ͼƬ�洢Ϊ[0, 1]����caffe�н�ͼƬ�洢Ϊ[0, 255]��������Ҫһ��ת��
                       image_dims=(224, 224),
                       mean=np.load(caffe_root +'icvl/mean.npy').mean(1).mean(1))#���þ�ֵ�ļ�) #����ģ�͵�ͼƬҪ��224*224��ͼƬ

#�����ǩ�ļ�
imagenet_labels_filename = caffe_root +'icvl/labels.txt'
#��������ǩ�ļ�
labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')

#��Ŀ��·���е�ͼ�񣬱���������
def asd():
    for root,dirs,files in os.walk('/mnt/5858379E583779B8/VSC/Python/Machine Learning/icvl/whiteres2/5/'):
        for file in files:
            #����Ҫ�����ͼƬ
            image_file = os.path.join(root,file)
            input_image = caffe.io.load_image(image_file)

            #��ӡͼƬ·��������
            image_path = os.path.join(root,file)
            print(image_path)
            
            #��ʾͼƬ
            img=Image.open(image_path)
            plt.imshow(img)
            plt.axis('off')
            plt.show()
            
            #Ԥ��ͼƬ���
            prediction = net.predict([input_image])
            print 'predicted class:',prediction[0].argmax()

            # �����������ǰ5��Ԥ����
            top_k = prediction[0].argsort()[-5:][::-1]
            for node_id in top_k:     
                #��ȡ��������
                human_string = labels[node_id]
                #��ȡ�÷�������Ŷ�
                score = prediction[0][node_id]
                print('%s (score = %.5f)' % (human_string, score))
if __name__ == "__main__":
    input_image = caffe.io.load_image('/mnt/5858379E583779B8/VSC/Python/Machine Learning/icvl/pose-ren/src/imgs/201912181395223520/gujia.png')
    pre = net.predict([input_image])
    print 'predicted class:',pre[0].argmax()
    asd()