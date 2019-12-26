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
                #获取分类名称
            label.append(node_id)
                #获取该分类的置信度
            score = pre[0][node_id]
            scores.append('%.5f' % score)

        print scores
        return render_template('success.html',filename=nt+'res.png',classres=str(pre[0].argmax()),label=label,scores=scores)


if __name__ == "__main__":
    app.run(port=5001)