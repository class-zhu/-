#coding=utf-8
from flask import Flask, flash, request, redirect, url_for,jsonify,render_template
import os
app = Flask(__name__)
import time
import onepre
from utils import util
        # f.close()
import caffe
import shutil
import numpy as np
from utils.model_pose_ren import ModelPoseREN
hand_model = ModelPoseREN('icvl')
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





def nowtime():
    t=time.localtime()
    res = ''
    for tt in t:
        res=res+str(tt)
    return res

@app.route('/',methods=['GET','POST'])
def preone():
    if(request.method=='GET'):
        return render_template('mpage.html')
    else:
        base_dir='/mnt/5858379E583779B8/VSC/Python/Machine Learning/icvl/pose-ren/src/imgs/'
        nt=nowtime()
        thisfile = request.files['file']
        os.makedirs('./imgs/'+nt)
        thisfile.save('./imgs/'+nt+'/'+thisfile.filename)
        f=open('./imgs/'+nt+'/name.txt','w')
        f.writelines(nt+'/'+thisfile.filename)
        f.close()
        print base_dir+nt+'/name.txt'
        onepre.pre(hand_model,base_dir,base_dir+nt+'/name.txt',base_dir+nt+'/')
        
        shutil.copy(base_dir+nt+'/res.png','./static/'+nt+"res.png")
        input_image = caffe.io.load_image(base_dir+nt+'/gujia.png')
        pre = net.predict([input_image])
        top_k = pre[0].argsort()[-5:][::-1]
        label = []
        scores = []
        for node_id in top_k:     
                #��ȡ��������
            label.append(node_id)
                #��ȡ�÷�������Ŷ�
            score = pre[0][node_id]
            scores.append('%.5f' % score)

        print scores
        return render_template('success.html',filename=nt+'res.png',classres=str(pre[0].argmax()),label=label,scores=scores)


if __name__ == "__main__":
    app.run(port=5001)