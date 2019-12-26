import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Slice(nn.Module):
    def __init__(self, axis, slice_points):
        super(Slice, self).__init__()
        self.axis = axis
        self.slice_points = slice_points

    def __repr__(self):
        return 'Slice(axis=%d, slice_points=%s)' % (self.axis, self.slice_points)

    def forward(self, x):
        prev = 0
        outputs = []
        is_cuda = x.data.is_cuda
        if is_cuda: device_id = x.data.get_device()
        for idx, slice_point in enumerate(self.slice_points):
            rng = range(prev, slice_point)
            rng = torch.LongTensor(rng)
            if is_cuda: rng = rng.cuda(device_id)
            rng = Variable(rng)
            y = x.index_select(self.axis, rng)
            prev = slice_point
            outputs.append(y)
        return tuple(outputs)


class Ren(nn.Module):

    def __init__(self):
        super(Ren, self).__init__()

        self.conv0 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv1 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv2_0 = nn.Conv2d(16, 32, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3_0 = nn.Conv2d(32, 64, 1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)

        self.slice_feat = Slice(2, [6, 12])
        self.slice_feat_1 = Slice(3, [6, 12])
        self.slice_feat_2 = Slice(3, [6, 12])

        self.fc1_1_1 = nn.Linear(64 * 6 * 6, 2048)
        self.fc2_1_1 = nn.Linear(2048, 2048)
        self.fc1_1_2 = nn.Linear(64 * 6 * 6, 2048)
        self.fc2_1_2 = nn.Linear(2048, 2048)
        self.fc1_2_1 = nn.Linear(64 * 6 * 6, 2048)
        self.fc2_2_1 = nn.Linear(2048, 2048)
        self.fc1_2_2 = nn.Linear(64 * 6 * 6, 2048)
        self.fc2_2_2 = nn.Linear(2048, 2048)

        self.fc_concat = nn.Linear(2048 * 4, 48)

        self.drop1_1_1 = nn.Dropout(0.5)
        self.drop2_1_1 = nn.Dropout(0.5)
        self.drop1_1_2 = nn.Dropout(0.5)
        self.drop2_1_2 = nn.Dropout(0.5)
        self.drop1_2_1 = nn.Dropout(0.5)
        self.drop2_2_1 = nn.Dropout(0.5)
        self.drop1_2_2 = nn.Dropout(0.5)
        self.drop2_2_2 = nn.Dropout(0.5)

    def forward(self, x):
        out = F.relu(self.conv0(x))
        out = F.max_pool2d(self.conv1(out), 2, 2)
        out = F.relu(out)

        # 残差连接1
        out = F.relu(self.conv2_0(out))
        residual1 = out
        out = F.relu(self.conv2(out))
        out = self.conv3(out)
        out = out + residual1
        out = F.max_pool2d(out, 2, 2)
        out = F.relu(out)

        # 残差连接2
        out = F.relu(self.conv3_0(out))
        residual2 = out
        out = F.relu(self.conv4(out))
        out = self.conv5(out)
        out = out + residual2
        out = F.max_pool2d(out, 2, 2)
        out = F.relu(out)

        # slice feature map
        '''
            full_size: 64 * 12 * 12
            each_region_size: 64 * 6 * 6
            ·-------·-------·
            |       |       |
            |  1_1  |  2_1  |  
            |       |       |
            ·-------·-------·
            |       |       |
            |  1_2  |  2_2  |   
            |       |       |
            ·-------·-------·
        '''
        outs = self.slice_feat(out)
        outs_1 = self.slice_feat_1(outs[0])
        outs_2 = self.slice_feat_2(outs[1])

        batch_size = len(outs_1[0])
        # 左上
        out_1_1 = outs_1[0].view(batch_size, 2304)
        out_1_1 = F.relu(self.fc1_1_1(out_1_1))
        out_1_1 = self.drop1_1_1(out_1_1)
        out_1_1 = F.relu(self.fc2_1_1(out_1_1))
        out_1_1 = self.drop2_1_1(out_1_1)

        # 左下
        out_1_2 = outs_1[1].view(batch_size, 2304)
        out_1_2 = F.relu(self.fc1_1_2(out_1_2))
        out_1_2 = self.drop1_1_2(out_1_2)
        out_1_2 = F.relu(self.fc2_1_2(out_1_2))
        out_1_2 = self.drop2_1_2(out_1_2)

        # 右上
        out_2_1 = outs_2[0].view(batch_size, 2304)
        out_2_1 = F.relu(self.fc1_2_1(out_2_1))
        out_2_1 = self.drop1_2_1(out_2_1)
        out_2_1 = F.relu(self.fc2_2_1(out_2_1))
        out_2_1 = self.drop2_2_1(out_2_1)

        # 右下
        out_2_2 = outs_2[1].view(batch_size, 2304)
        out_2_2 = F.relu(self.fc1_2_2(out_2_2))
        out_2_2 = self.drop1_2_2(out_2_2)
        out_2_2 = F.relu(self.fc2_2_2(out_2_2))
        out_2_2 = self.drop2_2_2(out_2_2)

        out = torch.cat((out_1_1, out_1_2, out_2_1, out_2_2), 1)

        return self.fc_concat(out)

    def set_weights(self, state_dict):
        self.load_state_dict(state_dict)


if __name__ == '__main__':
    # 测试网络可用性
    net = Ren()
    index = 0
    for name, param in list(net.named_parameters()):
        print(str(index) + ':', name, param.size())
        index += 1
