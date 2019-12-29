import cv2
import os

def picture2video(picture_path,newvideo_path,picture_amount):
    fps = 30  
    img_size = (774,258)
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G') #opencv3.0
    videoWriter = cv2.VideoWriter(newvideo_path, fourcc, fps, img_size)
    for i in range(0,picture_amount):
        im_name = os.path.join(picture_path, "pred_"+str(i)+'.jpg')
        frame = cv2.imread(im_name)
        videoWriter.write(frame)
        print(im_name)
    videoWriter.release()
    print("finish")

picture_path='/home/x/Documents/Code/Python/DeepLearning/Hand/hand-graph-cnn-master/output/configs/eval_real_world_testset.yaml'
newvideo_path = '/home/x/Documents/Code/Python/DeepLearning/Hand/hand-graph-cnn-master/output/configs/picture2video/newvideo_3.avi'
picture_amount=600
picture2video(picture_path,newvideo_path,picture_amount)