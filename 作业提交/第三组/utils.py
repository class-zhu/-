import cv2
import numpy as np

DATA_SET = 'icvl'
CUBE_SIZE = 150
INPUT_SIZE = 96
#fx, fy, ux, uy = 463.889, 463.889, 320, 240
LOWER = 0
UPPER = 500


fx, fy, ux, uy = 240.99, 240.96, 160, 120


def get_positions(in_file):
    with open(in_file) as f:
        positions = [list(map(float, line.strip().split())) for line in f]
    return np.reshape(np.array(positions), (-1, int(len(positions[0]) / 3), 3))


def check_dataset(dataset):
    return dataset in set(['icvl', 'nyu', 'msra'])


def get_dataset_file(dataset):
    return 'labels/{}_test_label.txt'.format(dataset)


def pixel2world(x, fx, fy, ux, uy):
    x[:, :, 0] = (x[:, :, 0] - ux) * x[:, :, 2] / fx
    x[:, :, 1] = (x[:, :, 1] - uy) * x[:, :, 2] / fy
    return x


def get_errors(dataset, in_file):
    if not check_dataset(dataset):
        print('invalid dataset: {}'.format(dataset))
        exit(-1)
    labels = get_positions(get_dataset_file(dataset))
    outputs = get_positions(in_file)
    params = 240.99, 240.96, 160, 120
    labels = pixel2world(labels, *params)
    outputs = pixel2world(outputs, *params)
    errors = np.sqrt(np.sum((labels - outputs) ** 2, axis=2))
    return errors


def load_image(name):
    img = cv2.imread(name, 2)  # depth image
    # img[img > 20] = 1
    print(img)
    print(img.max())
    img[img == 0] = 33000  # invalid pixel
    # img = img[:, ::-1]
    img = img.astype(float)

    return img


def transform_pose(poses, centers):
    res_poses = np.array(poses.detach().numpy()) * CUBE_SIZE
    num_joint = int(poses.shape[1] / 3)
    centers_tile = np.tile(centers, (num_joint, 1, 1)).transpose([1, 0, 2])
    res_poses[:, 0::3] = res_poses[:, 0::3] * fx / centers_tile[:, :, 2] + centers_tile[:, :, 0]
    res_poses[:, 1::3] = res_poses[:, 1::3] * fy / centers_tile[:, :, 2] + centers_tile[:, :, 1]
    res_poses[:, 2::3] += centers_tile[:, :, 2]
    res_poses = np.reshape(res_poses, [poses.shape[0], -1, 3])
    return res_poses


def get_sketch_setting():
    return [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
            (0, 7), (7, 8), (8, 9), (0, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15)]


def draw_pose(img, pose):
    for pt in pose:
        cv2.circle(img, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)
    for x, y in get_sketch_setting():
        cv2.line(img, (int(pose[x, 0]), int(pose[x, 1])),
                 (int(pose[y, 0]), int(pose[y, 1])), (0, 0, 255), 1)
    return img


def crop_image(img, center, is_debug=True):
    print(center)
    xstart = center[0] - CUBE_SIZE / center[2] * fx
    xend = center[0] + CUBE_SIZE / center[2] * fx
    ystart = center[1] - CUBE_SIZE / center[2] * fy
    yend = center[1] + CUBE_SIZE / center[2] * fy

    src = [(xstart, ystart), (xstart, yend), (xend, ystart)]
    dst = [(0, 0), (0, INPUT_SIZE - 1), (INPUT_SIZE - 1, 0)]
    trans = cv2.getAffineTransform(np.array(src, dtype=np.float32),
                                   np.array(dst, dtype=np.float32))
    res_img = cv2.warpAffine(img, trans, (INPUT_SIZE, INPUT_SIZE), None,
                             cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, center[2] + CUBE_SIZE)
    res_img -= center[2]
    res_img = np.maximum(res_img, -CUBE_SIZE)
    res_img = np.minimum(res_img, CUBE_SIZE)
    res_img /= CUBE_SIZE

    if is_debug:
        img_show = (res_img + 1) / 2
        hehe = cv2.resize(img_show, (512, 512))
        cv2.imshow('debug', img_show)
        ch = cv2.waitKey(33)
        if ch == ord('q'):
            exit(0)

    return res_img


def get_center(img, upper=UPPER, lower=LOWER):
    centers = np.array([0.0, 0.0, 300.0])
    count = 0
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if img[y, x] <= upper and img[y, x] >= lower:
                centers[0] += x
                centers[1] += y
                centers[2] += img[y, x]
                count += 1
    if count:
        centers /= count
    return centers


def generate_single_input(img, center=None):
    if center:
        return crop_image(img, center), center
    else:
        center = get_center(img)
        return crop_image(img, center), center


def show_pose(img, pose, wait_key=0):
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min()) * 255
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = draw_pose(img, pose)
    cv2.imshow('result', img / 255)
    cv2.waitKey(wait_key)
