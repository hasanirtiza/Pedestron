import numpy as np
import re
import json
import matplotlib.pyplot as plt

def parse_log_file(fname):
    with open(fname) as f:
        content = f.readlines()
    data = []
    num_pre = -1
    for line in content:
        if 'Epoch' in line and 'loss' in line and (not 'nan' in line):
            start = line.find('Epoch')
            start2 = line.find('time')
            # find all float number in string
            result1 = re.findall(r"[-+]?\d*\.\d+|\d+", line[start:start2])
            result2 = re.findall(r"[-+]?\d*\.\d+|\d+", line[start2:])
            result = result1[0:4] + result2
            assert num_pre < 0 or len(result)==num_pre, 'number of parse loss should be the same'
            data.append(np.array([float(item) for item in result]))
            num_pre = len(result)
    data = np.array(data)
    print(data.shape)

    iteration = (data[:,0]-1)*(data[:,2]) + data[:,1]
    lr_rate = data[:,3]

    # loss starts from index 10
    data = data[:,7:]

    plt.subplot(221)
    plt.plot(iteration, data[:, -1]) # total loss
    plt.subplot(222)
    plt.plot(iteration, data[:, -2]) # box loss
    plt.subplot(223)
    plt.plot(iteration, data[:, -4]) # class acc
    plt.subplot(224)
    plt.plot(iteration, lr_rate)
    plt.show()


if __name__ == '__main__':
    fname = '/home/hust/tools/log_fold/mmdetect/log_cascade_rcnn_x152_caffe_32x8d_fpn.txt'
    fname = '/home/hust/tools/log_fold/mmdetect/log_cascade_rcnn_x101_64x4d_fpn_1x_trnbn.txt'
    fname = '/home/hust/tools/log_fold/mmdetect/log_cascade_rcnn_densenet161.txt'
    parse_log_file(fname)