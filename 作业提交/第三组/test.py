from ren import Ren
import utils
from load_models import *
import torch
import cv2
import numpy as np



def get_center_fast(img, upper=500, lower=0):
    centers = np.array([0.0, 0.0, 300.0])
    flag = np.logical_and(img <= upper, img >= lower)
    x = np.linspace(0, img.shape[1], img.shape[1])
    y = np.linspace(0, img.shape[0], img.shape[0])
    xv, yv = np.meshgrid(x, y)
    centers[0] = np.mean(xv[flag])
    centers[1] = np.mean(yv[flag])
    centers[2] = np.mean(img[flag])
    if centers[2] <= 0:
        centers[2] = 300.0
    if not flag.any():
        centers[0] = 0
        centers[1] = 0
        centers[2] = 300.0
    return centers


def read_frame_from_device(dev):
    img_rgb = dev.colour
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    depth = dev.depth * dev.depth_scale * 1000
    return depth, img_rgb


if __name__ == '__main__':
    nw = load_weights_from_pkl(pkl_path)
    state_dict = get_torch_state_dict(nw)

    net = Ren()

    net.set_weights(state_dict)

    net.eval()

    # from_img = utils.load_image('img/td/3.png')
    from_img = utils.load_image('Depth/test_seq_2/image_0004.png')
    img, center = utils.generate_single_input(from_img)
    #center =[285,230,255]
    img = img.reshape(1, 1, 96, 96)
    img = torch.tensor(img, dtype=torch.float32)
    out = net.forward(img)
    res = utils.transform_pose(out, [center])

    utils.show_pose(from_img, res[0])
