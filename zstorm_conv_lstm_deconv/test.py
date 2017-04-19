from __future__ import print_function, division
import cv2
import numpy as np
import platform
from sklearn import metrics
import scipy.io as scio
import shelve
import argparse
import sys
import os
import progressbar

sys.path.append('/home/luowx/conv_lstm_master/python')
import caffe

T = 10
N = 1
new_height = 225
new_width = 225

testing_path_list = '/home/luowx/conv_lstm_master/zstorm_conv_lstm_deconv/testing_path_list.txt'
gt_path_list = '/home/luowx/conv_lstm_master/zstorm_conv_lstm_deconv/gt_path_list.txt'
mean_file_folder = '/home/luowx/datasets/'
deploy_folder = '/home/liuwen/new_disk1/zstorm/32/'

def parser_args():
    '''
    python test.py --gpu 1 \
                   --datasets 'enter_authors' \
                   --deploy /home/liuwen/conv_lstm/zstorm_conv_lstm_deconv_enter/deploy_enter.prototxt \
                   --mean /home/liuwen/new_disk1/datasets/original/enter_authors_mean_225_gray.npy \
                   --caffemodel /home/liuwen/conv_lstm/zstorm_conv_lstm_deconv_enter/snapshot/zstorm_conv_lstm_deconv_current_past_enter_iter_90000.caffemodel \
                   --channels 1 \
                   --listening 0


    python test.py --gpu 1 \
               --datasets 'ped2' \
               --deploy /new_disk1/liuwen/zstorm/32/deploy_conv_lstm_deconv_new_T50.prototxt \
               --mean /home/liuwen/new_disk1/datasets/original/ped2_mean_225_gray.npy \
               --caffemodel caffemodel /home/liuwen/conv_lstm/zstorm_conv_lstm_deconv_full/ped2/ \
               --channels 1 \
               --listening 0

    :return:
    '''
    parser = argparse.ArgumentParser(description='converting npy file to binaryproto file, '
                                                 'shape is channels x height x width...')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device ID')
    parser.add_argument('--datasets', type=str, default=['avenue', 'ped1', 'ped2', 'enter', 'exit'], nargs='+', help='choosing the datasets.')
    parser.add_argument('--mean', type=str, help='mean.npy')
    parser.add_argument('--deploy', type=str, help='deploy.prototxt')
    parser.add_argument('--caffemodel', type=str, help='caffemodel')
    parser.add_argument('--channels', type=int, default=1, help='the number of channels.')
    parser.add_argument('--listening', type=int, default=0, help='set is listening or not, 1 is listening, 0 is not.')
    return parser.parse_args()


def parser_file_list(testing_path_list, gt_path_list):
    testing_data = {}
    with open(testing_path_list, 'r') as f:
        testing_video_path_list = f.readlines()
    with open(gt_path_list, 'r') as f:
        ground_truth_path_list = f.readlines()

    assert len(testing_video_path_list) > 0, 'there is no any testing data in path {}'.format(testing_path_list)
    assert len(ground_truth_path_list) > 0, 'there is no any testing data ground truth in path {}'.format(gt_path_list)

    for gt_path in ground_truth_path_list:
        # gt_path: /home/liuwen/new_disk1/datasets/original/avenue.mat \n
        gt_path = gt_path.rstrip()
        data_set_name = gt_path.split('/')[-1].split('.')[0]
        if data_set_name not in testing_data:
            testing_data[data_set_name] = dict()
            testing_data[data_set_name]['gt_path'] = gt_path
            testing_data[data_set_name]['tf_path'] = []
        else:
            raise Exception('there has been a testing data set named {}'.format(data_set_name))

        for tf_path in testing_video_path_list:
            # tf_path: /home/liuwen/new_disk1/datasets/original/avenue/testing_videos/01.avi \n
            tf_path = tf_path.rstrip()
            testing_video_name = tf_path.split('/')[-3]
            if data_set_name == testing_video_name:
                testing_data[data_set_name]['tf_path'].append(tf_path)
            else:
                continue
    return testing_data


def calcDistance(x, y):
    return np.linalg.norm(x - y)


def calcSquareDistance(x, y):
    return np.sum(np.square(x - y))


def crop(image, crop_size):
    '''
    :param image: C x H x W
    :param crop_size: [crop_h, crop_w]
    :return: C x H x W
    '''
    assert image.shape[1] >= crop_size[0] and image.shape[2] >= crop_size[1], \
        'crop size must be smaller than original image.'
    h_off = (image.shape[1] - crop_size[0]) / 2
    w_off = (image.shape[2] - crop_size[1]) / 2
    return image[:, h_off:crop_size[0] + h_off, w_off:crop_size[1] + w_off]


def transform(image, mean, crop_size, is_sub_mean=True, is_scale=True, is_transpose=True):
    '''
    :param image: H x W x C
    :param mean: H x W x C
    :param crop_size: [crop_h, crop_w]
    :return:
    '''

    channels = image.shape[2]
    image = image.astype(np.float32, copy=False)
    image = cv2.resize(image, (new_height, new_width))
    image = np.reshape(image, (new_height, new_width, channels))


    # image [0, 255]
    # mean [0, 255]
    image = (image - mean) / 255.0

    if is_transpose:
        image = image.transpose((2, 0, 1)) # C x H x W
    # image = crop(image, crop_size) # C x crop_h x crop_w
    return image


def load_video3input(capture, length, height, width, channels, mean_image):
    # progress bar
    bar = progressbar.ProgressBar(maxval=length,
                                  widgets=[progressbar.Bar('>', '[', ']'), ' ', progressbar.SimpleProgress(), ' ',
                                           progressbar.Percentage(), ' ', progressbar.ETA()]).start()
    # frame_volume[t, 0] = t, t-1, t-2
    frame_volume = np.zeros((length, 1, 3, height, width), dtype=np.float32)


    for i in range(0, length):
        bar.update(i + 1)
        retval, frame = load_convert_image(capture, mean_image, height, width, channels,
                                               is_sub_mean=True,
                                               is_scale=True,
                                               is_transpose=True)
        if retval:
            if i == 0:
                frame_volume[0, 0, 0, ...] = frame      # frame_volume[0] = t0, t0, t1
                frame_volume[0, 0, 1, ...] = frame
                frame_volume[1, 0, 0, ...] = frame      # frame_volume[1] = t0, x, x
            elif i == length - 1:
                frame_volume[length - 1, 0, 1, ...] = frame # frame_volume[length - 1]
                                                                            # = t[length-2], t[length-2], t[length - 1]
                frame_volume[length - 1, 0, 2, ...] = frame
                frame_volume[length - 2, 0, 2, ...] = frame # frame_volume[length - 2] = x, x, t[length - 1]
            else:
                frame_volume[i - 1, 0, 2, ...] = frame      # frame_volume[i - 1] = x, x, t[i]
                frame_volume[i + 1, 0, 0, ...] = frame      # frame_volume[i + 1] = t[i], x, x

                frame_volume[i, 0, 1, ...] = frame          # frame_volume[i] = x, t[i], x
        else:
            frame_volume[i] = frame_volume[i - 1]
    bar.finish()

    return frame_volume


def load_video2images(capture, length, height, width, channels, mean_image):
    # progress bar
    bar = progressbar.ProgressBar(maxval=length,
                                  widgets=[progressbar.Bar('>', '[', ']'), ' ', progressbar.SimpleProgress(), ' ',
                                           progressbar.Percentage(), ' ', progressbar.ETA()]).start()
    frame_volume = np.zeros((length, 1, channels, height, width), dtype=np.float32)

    for i in range(0, length):
        # bar.update(i)
        retval, frame = load_convert_image(capture, mean_image, height, width, channels,
                                           is_sub_mean=True,
                                           is_scale=True,
                                           is_transpose=True)
        if retval:
            frame_volume[i, 0, ...] = frame
        else:
            frame_volume[i] = frame_volume[i - 1]
            print('missing frame {}'.format(i))

    capture.release()
    return frame_volume


def roc(pred, label):
    fpr, tpr, thresholds = metrics.roc_curve(label, pred, pos_label=0)
    return metrics.auc(fpr, tpr)


def load_convert_image(capture, mean_image, height, width, channels, is_sub_mean=True, is_scale=True, is_transpose=True):
    retval, image = capture.read()  # BGR
    if not retval:
        transformed_image = None
    else:
        if channels == 1:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = np.reshape(image, (image.shape[0], image.shape[1], 1))
        transformed_image = transform(image, mean_image, (height,width), is_sub_mean, is_scale, is_transpose)  # C*H*W   BGR

    return retval, transformed_image


def refer_2loss_overlap(net, caffemodel_name, mean_file, data_set_name, tf_path, gt_path, channels=1, loss_type='1-loss'):
    mean_image = np.load(mean_file)
    gt = scio.loadmat(gt_path, squeeze_me=True)['gt']
    print(gt.shape)
    score = np.array([], dtype=np.float32)
    label = np.array([], dtype=np.int32)
    total_distance = np.array([], dtype=np.float32)

    if data_set_name == 'exit' or data_set_name == 'enter_authors':
        mask = np.ones((225, 225, 1), dtype=np.uint8)
        mask[165: 195, 125: 215, 0] = 0
    else:
        mask = np.ones((225, 225, 1), dtype=np.uint8)

    height = net.blobs['data'].data.shape[3]
    width = net.blobs['data'].data.shape[4]
    frame_volumn = np.zeros((T, 1, 1, height, width), dtype=np.float32)

    for (i, video_name) in enumerate(tf_path):
        # video_name: /home/liuwen/new_disk1/datasets/original/avenue/testing_videos/01.avi
        video_idx = int(video_name.split('/')[-1].split('.')[0]) - 1
        name = data_set_name + '/' + video_name.split('/')[-1]
        assert i == video_idx, 'video id {} is different from index {}'.format(video_idx, i)
        print('\tcomputing video: {}'.format(video_name))
        print('\t{}'.format(caffemodel_name))

        capture = cv2.VideoCapture(video_name.strip('\n'))
        length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        video_distance = np.zeros(length, dtype=np.float32)
        video_label = np.zeros(length, dtype=np.float32)

        _T = min(T, length)
        # progress bar
        bar = progressbar.ProgressBar(maxval=length,
                                      widgets=[progressbar.Bar('>', '[', ']'), ' ', progressbar.SimpleProgress(), ' ',
                                               progressbar.Percentage(), ' ', progressbar.ETA()]).start()

        net.blobs['cont'].data[...] = 1
        net.blobs['cont'].data[0] = 0

        # pre-processing [0, T] frames
        for j in range(0, _T):
            bar.update(j + 1)
            retval, transformed_image = load_convert_image(capture, mean_image, height, width, channels,
                                                           is_sub_mean=True,
                                                           is_scale=True,
                                                           is_transpose=True)
            if retval:
                frame_volumn[j] = transformed_image
            else:
                frame_volumn[j] = frame_volumn[j - 1]

        net.blobs['data'].data[...] = frame_volumn
        output = net.forward()

        #video_distance[0] = calcDistance(net.blobs['data'].data[0], output['deconv5_1'][0]) * 2
        for k in range(int(_T / 2), _T):
            video_distance[k] = np.linalg.norm((net.blobs['data'].data[k] - output['deconv3_1'][k]) * mask) ** 2 \
                                        + np.linalg.norm((net.blobs['data'].data[k - 1] - output['deconv3_2'][k - 1]) * mask) ** 2
            video_distance[k - int(_T / 2)] = video_distance[k]

        # processing [T, length]
        T_sub = int(_T / 2) # T_sub = 10 / 2
        for j in range(_T, length):
            bar.update(j + 1)
            retval, transformed_image = load_convert_image(capture, mean_image, height, width, channels,
                                                           is_sub_mean=True,
                                                           is_scale=True,
                                                           is_transpose=True)
            if retval:
                frame_volumn[j % T_sub] = transformed_image
            else:
                frame_volumn[j % T_sub] = frame_volumn[(j - 1) % T_sub]

            if j % T_sub == T_sub - 1 or j == length - 1:
                T_sub_left = j % T_sub
                net.blobs['data'].data[0: T_sub] = frame_volumn[T_sub: _T]
                net.blobs['data'].data[T_sub: T_sub + T_sub_left + 1] = frame_volumn[0: T_sub_left + 1]
                output = net.forward()

                for k in range(j - T_sub_left, j + 1):
                    d_k = T_sub + k % T_sub
                    video_distance[k] = np.linalg.norm((net.blobs['data'].data[d_k] - output['deconv3_1'][d_k]) * mask) ** 2 \
                                        + np.linalg.norm((net.blobs['data'].data[d_k - 1] - output['deconv3_2'][d_k - 1]) * mask) ** 2

                frame_volumn[T_sub: _T] = frame_volumn[0: T_sub]

        bar.finish()

        if gt.ndim == 2:
            gt = gt.reshape(-1, gt.shape[0], gt.shape[1])
        gt[i] = np.reshape(gt[i], (2, -1))
        for j in range(gt[i].shape[1]):
            for k in range(gt[i][0, j] - 1, gt[i][1, j]):
                video_label[k] = 1

        video_distance = np.sqrt(video_distance)

        # np.savetxt('distances/' + str(video_idx), video_distance)
        total_distance = np.append(total_distance, video_distance)
        video_distance -= video_distance.min()
        video_distance /= video_distance.max()
        video_distance = 1 - video_distance
        score = np.append(score, video_distance)
        label = np.append(label, video_label)

    label_txt = np.loadtxt('label_txt/' + data_set_name + '.txt', dtype=np.uint8)
    assert np.count_nonzero(label == label_txt) == len(label), 'label is different with label_txt'

    return score, label, total_distance


def refer_future(net, caffemodel_name, mean_file, data_set_name, tf_path, gt_path, channels=1, loss_type='1-loss'):
    mean_image = np.load(mean_file)
    gt = scio.loadmat(gt_path, squeeze_me=True)['gt']
    print(gt.shape)
    score = np.array([], dtype=np.float32)
    label = np.array([], dtype=np.int32)
    total_distance = np.array([], dtype=np.float32)

    if data_set_name == 'exit' or data_set_name == 'enter_authors':
        mask = np.ones((225, 225, 1), dtype=np.uint8)
        mask[165: 195, 125: 215, 0] = 0
    else:
        mask = np.ones((225, 225, 1), dtype=np.uint8)

    height = net.blobs['data'].data.shape[3]
    width = net.blobs['data'].data.shape[4]

    for (i, video_name) in enumerate(tf_path):
        # video_name: /home/liuwen/new_disk1/datasets/original/avenue/testing_videos/01.avi
        video_idx = int(video_name.split('/')[-1].split('.')[0]) - 1
        name = data_set_name + '/' + video_name.split('/')[-1]
        assert i == video_idx, 'video id {} is different from index {}'.format(video_idx, i)
        print('\tcomputing video: {}'.format(video_name))
        print('\t{}'.format(caffemodel_name))

        capture = cv2.VideoCapture(video_name.strip('\n'))
        length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        video_distance = np.zeros(length, dtype=np.float32)
        video_label = np.zeros(length, dtype=np.float32)

        video_frames = load_video2images(capture, length, height, width, channels, mean_image)

        _T = min(T, length)
        # progress bar
        bar = progressbar.ProgressBar(maxval=length,
                                      widgets=[progressbar.Bar('>', '[', ']'), ' ', progressbar.SimpleProgress(), ' ',
                                               progressbar.Percentage(), ' ', progressbar.ETA()]).start()

        net.blobs['cont'].data[...] = 1
        net.blobs['cont'].data[0] = 0

        j = 0
        while j < length:
            bar.update(j)

            start = j
            end = start + T

            # print(start, end)
            net.blobs['data'].data[...] = video_frames[start: end]
            output = net.forward()

            for k in range(start, end):
                if k == length - 1:
                    video_distance[k] = np.linalg.norm((video_frames[k] - output['deconv3'][k - start]) * mask) ** 2
                else:
                    video_distance[k] = np.linalg.norm((video_frames[k + 1] - output['deconv3'][k - start]) * mask) ** 2

            j += T

        bar.finish()

        if gt.ndim == 2:
            gt = gt.reshape(-1, gt.shape[0], gt.shape[1])
        gt[i] = np.reshape(gt[i], (2, -1))
        for j in range(gt[i].shape[1]):
            for k in range(gt[i][0, j] - 1, gt[i][1, j]):
                video_label[k] = 1

        video_distance = np.sqrt(video_distance)

        # np.savetxt('distances/' + str(video_idx), video_distance)
        total_distance = np.append(total_distance, video_distance)
        video_distance -= video_distance.min()
        video_distance /= video_distance.max()
        video_distance = 1 - video_distance
        score = np.append(score, video_distance)
        label = np.append(label, video_label)

    label_txt = np.loadtxt('label_txt/' + data_set_name + '.txt', dtype=np.uint8)
    assert np.count_nonzero(label == label_txt) == len(label), 'label is different with label_txt'

    return score, label, total_distance


'''
python test.py --gpu 1 \
               --datasets 'enter_authors' \
               --deploy /home/liuwen/conv_lstm/zstorm_conv_lstm_deconv_enter/deploy_enter.prototxt \
               --mean /home/liuwen/new_disk1/datasets/original/enter_authors_mean_225_gray.npy \
               --caffemodel /home/liuwen/conv_lstm/zstorm_conv_lstm_deconv_enter/snapshot/zstorm_conv_lstm_deconv_current_past_enter_iter_50000.caffemodel \
               --channels 1 \
               --listening 0

python test.py --gpu 2 \
               --datasets 'enter_authors' \
               --deploy /home/liuwen/new_disk1/zstorm_conv_lstm_deconv_full/deploy_T.prototxt \
               --mean /home/liuwen/new_disk1/datasets/original/enter_authors_mean_225_gray.npy \
               --caffemodel /home/liuwen/new_disk1/zstorm_conv_lstm_deconv_full/enter_authors/ \
               --channels 1 \
               --listening 0

python test.py --gpu 2 \
           --datasets 'ped1' \
           --deploy /home/liuwen/new_disk1/zstorm_conv_lstm_deconv_full/deploy_T.prototxt \
           --mean /home/liuwen/new_disk1/datasets/original/ped1_mean_225_gray.npy \
           --caffemodel /home/liuwen/new_disk1/zstorm_conv_lstm_deconv_full/ped1/ \
           --channels 1 \
           --listening 0

python test.py --gpu 2 \
           --datasets 'avenue' \
           --deploy /home/liuwen/new_disk1/zstorm_conv_lstm_deconv_full/deploy_T.prototxt \
           --mean /home/liuwen/new_disk1/datasets/original/avenue_mean_225_gray.npy \
           --caffemodel /home/liuwen/new_disk1/zstorm_conv_lstm_deconv_full/avenue/ \
           --channels 1 \
           --listening 0


python test.py --gpu 1 \
           --datasets 'ped2' \
           --deploy /home/liuwen/new_disk1/zstorm_conv_lstm_deconv_full/deploy_T.prototxt \
           --mean /home/liuwen/new_disk1/datasets/original/ped2_mean_225_gray.npy \
           --caffemodel /home/liuwen/new_disk1/zstorm_conv_lstm_deconv_full/ped2/ \
           --channels 1 \
           --listening 0


python test.py --gpu 1 \
           --datasets 'exit' \
           --deploy /home/liuwen/new_disk1/zstorm_conv_lstm_deconv_full/deploy_T.prototxt \
           --mean /home/liuwen/new_disk1/datasets/original/exit_mean_225_gray.npy \
           --caffemodel /home/liuwen/new_disk1/zstorm_conv_lstm_deconv_full/exit/ \
           --channels 1 \
           --listening 0

----------------------------- 1conv_lstm, 1-part ----------------------------------------------
python test.py --gpu 2 \
           --datasets 'exit' \
           --deploy /home/liuwen/new_disk1/zstorm/32/deploy_conv_lstm_deconv_new_T30.prototxt \
           --mean /home/liuwen/new_disk1/datasets/original/exit_mean_225_gray.npy \
           --caffemodel /home/liuwen/new_disk1/zstorm/32/exit/conv_lstm_deconv_T30/ \
           --channels 1 \
           --listening 0

python test.py --gpu 0 \
           --datasets 'ped1' \
           --deploy /home/liuwen/new_disk1/zstorm/32/deploy_conv_lstm_deconv_new_T2.prototxt \
           --mean /home/liuwen/new_disk1/datasets/original/ped1_mean_225_gray.npy \
           --caffemodel /home/liuwen/new_disk1/zstorm/32/ped1/conv_lstm_deconv_new_T2/ \
           --channels 1 \
           --listening 0

python test.py --gpu 0 \
           --datasets 'ped2' \
           --deploy /home/liuwen/new_disk1/zstorm/32/deploy_conv_lstm_deconv_new_T60.prototxt \
           --mean /home/liuwen/new_disk1/datasets/original/ped2_mean_225_gray.npy \
           --caffemodel /home/liuwen/new_disk1/zstorm/32/ped2/conv_lstm_deconv_new_T60/ \
           --channels 1 \
           --listening 0

python test.py --gpu 0 \
           --datasets 'enter_authors' \
           --deploy /home/liuwen/new_disk1/zstorm/32/deploy_conv_lstm_deconv_new_T10.prototxt \
           --mean /home/liuwen/new_disk1/datasets/original/enter_authors_mean_225_gray.npy \
           --caffemodel /home/liuwen/new_disk1/zstorm/32/enter_authors/conv_lstm_deconv_new_T10/ \
           --channels 1 \
           --listening 0

python test.py --gpu 1 \
           --datasets 'enter_authors' \
           --deploy /home/liuwen/new_disk1/zstorm/32/deploy_conv_lstm_deconv_new_T50.prototxt \
           --mean /home/liuwen/new_disk1/datasets/original/enter_authors_mean_225_gray.npy \
           --caffemodel /home/liuwen/new_disk1/zstorm/32/enter_authors/conv_lstm_deconv_new_T50/ \
           --channels 1 \
           --listening 0

-------------------------------------- rebuttal ped2 ----------------------------------------------
python test.py --gpu 0 \
            --datasets 'ped2' \
            --deploy /home/liuwen/PycharmProjects/zstorm/rebuttal/deploy_current_past_ped2_new_conv_deconv_4.prototxt \
            --mean /home/liuwen/new_disk1/datasets/original/ped2_mean_225_gray.npy \
            --caffemodel /home/liuwen/new_disk1/zstorm/32/rebuttal/conv_deconv_4/ \
            --channels 1 \
            --listening 0

python test.py --gpu 1 \
            --datasets 'ped2' \
            --deploy /home/liuwen/PycharmProjects/zstorm/rebuttal/deploy_current_past_ped2_new_conv_deconv_5.prototxt \
            --mean /home/liuwen/new_disk1/datasets/original/ped2_mean_225_gray.npy \
            --caffemodel /home/liuwen/new_disk1/zstorm/32/rebuttal/conv_deconv_5/ \
            --channels 1 \
            --listening 0

python test.py --gpu 2 \
            --datasets 'ped2' \
            --deploy /home/liuwen/PycharmProjects/zstorm/rebuttal/deploy_current_past_ped2_new_convlstm2.prototxt \
            --mean /home/liuwen/new_disk1/datasets/original/ped2_mean_225_gray.npy \
            --caffemodel /home/liuwen/new_disk1/zstorm/32/rebuttal/convlstm2/ \
            --channels 1 \
            --listening 0

python test.py --gpu 3 \
            --datasets 'ped2' \
            --deploy /home/liuwen/PycharmProjects/zstorm/rebuttal/deploy_current_past_ped2_new_convlstm3.prototxt \
            --mean /home/liuwen/new_disk1/datasets/original/ped2_mean_225_gray.npy \
            --caffemodel /home/liuwen/new_disk1/zstorm/32/rebuttal/convlstm3/ \
            --channels 1 \
            --listening 0

-------------------------------------- rebuttal ped2 ----------------------------------------------

python test.py --gpu 0 \
            --datasets 'ped1' \
            --deploy /home/liuwen/PycharmProjects/zstorm/rebuttal/deploy_current_past_ped2_new_conv_deconv_4.prototxt \
            --mean /home/liuwen/new_disk1/datasets/original/ped1_mean_225_gray.npy \
            --caffemodel /home/liuwen/new_disk1/zstorm/32/rebuttal/current_past/ped1/conv_deconv_4/ \
            --channels 1 \
            --listening 0

python test.py --gpu 1 \
            --datasets 'ped1' \
            --deploy /home/liuwen/PycharmProjects/zstorm/rebuttal/deploy_current_past_ped2_new_conv_deconv_5.prototxt \
            --mean /home/liuwen/new_disk1/datasets/original/ped1_mean_225_gray.npy \
            --caffemodel /home/liuwen/new_disk1/zstorm/32/rebuttal/current_past/ped1/conv_deconv_5/ \
            --channels 1 \
            --listening 0

python test.py --gpu 2 \
            --datasets 'ped1' \
            --deploy /home/liuwen/PycharmProjects/zstorm/rebuttal/deploy_current_past_ped2_new_convlstm2.prototxt \
            --mean /home/liuwen/new_disk1/datasets/original/ped1_mean_225_gray.npy \
            --caffemodel /home/liuwen/new_disk1/zstorm/32/rebuttal/current_past/ped1/convlstm2/ \
            --channels 1 \
            --listening 0

python test.py --gpu 3 \
            --datasets 'ped1' \
            --deploy /home/liuwen/PycharmProjects/zstorm/rebuttal/deploy_current_past_ped2_new_convlstm3.prototxt \
            --mean /home/liuwen/new_disk1/datasets/original/ped1_mean_225_gray.npy \
            --caffemodel /home/liuwen/new_disk1/zstorm/32/rebuttal/current_past/ped1/convlstm3/ \
            --channels 1 \
            --listening 0

-------------------------------------- rebuttal prediction future ped1 ----------------------------------------------
python test.py --gpu 3 \
            --datasets 'ped1' \
            --deploy /home/liuwen/PycharmProjects/zstorm/rebuttal/deploy_train_future_ped1_new.prototxt \
            --mean /home/liuwen/new_disk1/datasets/original/ped1_mean_225_gray.npy \
            --caffemodel /home/liuwen/new_disk1/zstorm/32/rebuttal/future/ped1/ \
            --channels 1 \
            --listening 0

-------------------------------------- rebuttal prediction ped2 ----------------------------------------------
python test.py --gpu 2 \
            --datasets 'ped2' \
            --deploy /home/liuwen/PycharmProjects/zstorm/rebuttal/deploy_train_future_ped2_new.prototxt \
            --mean /home/liuwen/new_disk1/datasets/original/ped2_mean_225_gray.npy \
            --caffemodel /home/liuwen/new_disk1/zstorm/32/rebuttal/future/ped2/ \
            --channels 1 \
            --listening 0
'''


def testing_model(caffemodel, testing_data, options):
    if options.gpu == -1:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(options.gpu)

    deploy = options.deploy
    print(deploy)
    print(caffemodel)

    # sys.exit(0)
    #time.sleep(10)

    net = caffe.Net(deploy, caffemodel, caffe.TEST)
    caffemodel_name = caffemodel.split('.caffemodel')[0]
    datasets = options.datasets
    db = shelve.open(caffemodel_name)
    for data_set_name in datasets:
        mean_file = options.mean
        value = testing_data[data_set_name]
        print('dataset: {}'.format(data_set_name))
        print('\tmean file: {}'.format(mean_file))
        print('\tground truth: {}'.format(value['gt_path']))
        print('\tvideo path: ')
        value['tf_path'].sort()

        score, label, video_distance_list = refer_2loss_overlap(net, caffemodel_name, mean_file, data_set_name, value['tf_path'], value['gt_path'],
                                               options.channels)
        print(score.shape, label.shape)
        auc = roc(score, label)
        db[data_set_name] = {'score': score, 'auc': auc, 'label': label, 'distance': video_distance_list}
        print('\t{}'.format(caffemodel_name))
        print('\tauc: {}'.format(auc))
    db.close()


def test(options):
    testing_data = parser_file_list(testing_path_list, gt_path_list)

    if os.path.isdir(options.caffemodel):
        caffemodel_list = []
        for caffemodel in os.listdir(options.caffemodel):
            if caffemodel.endswith('.caffemodel'):
                caffemodel_list.append(options.caffemodel + caffemodel)
    else:
        caffemodel_list = [options.caffemodel]

    caffemodel_list.sort()
    for caffemodel in caffemodel_list:
        testing_model(caffemodel, testing_data, options)

import threading
import Queue
import time


class Producer(threading.Thread):
    '''
    Producer Thread
    '''
    def __init__(self, thread_name, queue, caffemoel_folder):
        self.queue = queue
        self.is_run = True
        self.folder = caffemoel_folder
        self.used_sets_path = self.folder + 'model_sets'
        self.counter = 0
        threading.Thread.__init__(self, name=thread_name)

    def terminate(self):
        self.is_run = False

    def run(self):
        print('{} is running.'.format(self.name))
        used_model = shelve.open(self.used_sets_path, writeback=True)
        if 'model_sets' not in used_model:
            used_model['model_sets'] = set()
        model_sets = used_model['model_sets']

        while True and self.is_run:
            print('scanning for {}'.format(self.folder))
            add_model_set = self._scanning_folder() - model_sets
            model_sets |= add_model_set
            for add_model in add_model_set:
                self.queue.put(add_model)
                print('adding new model: {}'.format(add_model))
            used_model['model_sets'] = model_sets

            self.counter += 1
            if self.counter % 10 == 0:
                self.counter = 0
                used_model.sync()

            time.sleep(60)


    def _scanning_folder(self):
        current_model_sets = set([self.folder + _file for _file in os.listdir(self.folder) if _file.endswith('.caffemodel')])
        return current_model_sets


class Consumer(threading.Thread):
    def __init__(self, thread_name, queue, testing_data, options):
        self.queue = queue
        self.is_run = True
        self.testing_data = testing_data
        self.options = options
        threading.Thread.__init__(self, name=thread_name)

    def run(self):
        print('{} is running.'.format(self.name))
        while True and self.is_run:
            try:
                top_model = self.queue.get()
                testing_model(top_model, self.testing_data, self.options)
            except Exception, e:
                print(e.message)

    def terminate(self):
        self.is_run = False

def listening(options):
    assert os.path.isdir(options.caffemodel), '{} is not a directory.'.format(options.caffemodel)

    testing_data = parser_file_list(testing_path_list, gt_path_list)
    que = Queue.Queue()
    producer = Producer('producer', que, options.caffemodel)
    consumer = Consumer('consumer', que, testing_data, options)
    producer.start()
    consumer.start()
    producer.join()
    consumer.join()


if __name__ == '__main__':
    # parser_file_list(options.testing_path_list, options.gt_path_list)
    options = parser_args()
    if options.listening:
        listening(options)
    else:
        test(options)
