from ren import Ren
import utils
from load_models import *
import torch
import numpy as np
import os
import time

nw = load_weights_from_pkl(pkl_path)
state_dict = get_torch_state_dict(nw)
net = Ren()
net.set_weights(state_dict)

BASE_DIR_1 = 'Depth/test_seq_1'
BASE_DIR_2 = 'Depth/test_seq_2'
SEQ_SIZE = [702, 894]


def load_imgs(base_dir='Depth/test_seq_1', img_format="image_{}.png", batch_size=64, start=0):
    origin_imgs = []
    imgs = []
    centers = []
    for idx in range(batch_size):
        idx = "0" * (4 - len(str(idx + start))) + str(idx + start)
        from_img = utils.load_image(os.path.join(base_dir, img_format.format(idx)))
        origin_imgs.append(from_img)
        img, center = utils.generate_single_input(from_img)
        img = img.reshape(1, 96, 96)
        imgs.append(img)
        centers.append(center)

    return origin_imgs, np.array(imgs), centers


def detect_imgs(imgs, centers):
    net.eval()
    imgs = torch.tensor(imgs, dtype=torch.float32)
    outs = net.forward(imgs)
    res = utils.transform_pose(outs, centers)
    return res


def detect_single_img(img):
    img, center = utils.generate_single_input(img)
    img = img.reshape(1, 1, 96, 96)
    img = torch.tensor(img, dtype=torch.float32)
    out = net.forward(img)
    res = utils.transform_pose(out, [center])

    return res


def dump_result_to_line(result):
    res = result.reshape(result.size)
    line = " ".join(map(lambda x: format(x, '.3f'), res))
    print(line)
    return line


def get_all_test_results(base_dirs=[], seq_size=[], batch_size=64):
    results = []
    for idx in range(len(base_dirs)):
        print("predicting {} ......".format(base_dirs[idx]))
        start = 0
        remain = seq_size[idx]
        bs = batch_size
        while remain > 0:
            if remain < batch_size:
                bs = remain
            origin, inputs, centers = load_imgs(base_dir=base_dirs[idx], batch_size=bs, start=start)
            res = detect_imgs(inputs, centers)
            for r in res:
                results.append(dump_result_to_line(r))
            remain -= bs
            start += bs
            print(remain)
            print(start)

    print(len(results))
    return results


if __name__ == '__main__':
    start = time.time()
    results = get_all_test_results([BASE_DIR_1, BASE_DIR_2], SEQ_SIZE)
    end = time.time()
    with open("results/result_icvl_4x6x6.txt", 'w') as f:
        for result in results:
            f.write(result + '\n')
    print("预测完成，用时{:.2f}s".format(end - start))
