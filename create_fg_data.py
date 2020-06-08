import cv2
import numpy as np
import os
import glob
import pickle
import random
import cv2

class getFGs():
    """ Produce images read from a list of files. """

    def __init__(self, m_files, bg_dir, shape = 256, channel=3):
        """
        Args:
            files (list): list of file paths.
            channel (int): 1 or 3. Will convert grayscale to RGB images if channel==3.
                Will produce (h, w, 1) array if channel==1.
            resize (tuple): int or (h, w) tuple. If given, resize the image.
        """
        assert len(m_files)
        self.m_files = m_files
        self.bg_dir = bg_dir
        self.channel = int(channel)
        assert self.channel in [1, 3], self.channel
        self.imread_mode = cv2.IMREAD_GRAYSCALE if self.channel == 1 else cv2.IMREAD_COLOR
        self.maxsize = shape

    def __len__(self):
        return len(self.m_files)

    def __iter__(self):

        for j in self.m_files:
            if 'pedestrian' in j:
                filename = "_".join(os.path.basename(j).split('_')[:-1]) + '.png'
            else:
                filename = os.path.basename(j).split('_')[0] + '.jpg'
            f = os.path.join(self.bg_dir, filename)
            im_orig = cv2.imread(f, self.imread_mode)
            if 'MHP' in j:
                maski = np.array(cv2.imread(j))
                gray = cv2.cvtColor(maski, cv2.COLOR_BGR2GRAY)
                m = (gray > 0) * 255.
            else:
                m = cv2.imread(j, cv2.IMREAD_GRAYSCALE)

            assert im_orig is not None
            assert m is not None
            # im_orig = im_orig[:, :, ::-1]

            box = self.find_bbx(m)
            buffer = np.zeros_like(m)
            buffer[box[0]:box[2], box[1]:box[3]] = 1
            buffer = cv2.resize(buffer, (self.maxsize, self.maxsize))
            box_ = np.array([0, 0, 0, 0])
            xs = np.nonzero(np.sum(buffer, axis=0))[0]
            ys = np.nonzero(np.sum(buffer, axis=1))[0]
            box_[1] = xs.min()
            box_[3] = xs.max()
            box_[0] = ys.min()
            box_[2] = ys.max()

            im = im_orig[box[0]:box[2], box[1]:box[3], :]
            m = m[box[0]:box[2], box[1]:box[3]]/255.
            m = m[:,:, None]
            im= im*m



            yield im

    def find_bbx(self, maskj):
        resized_width = self.maxsize
        resized_height = self.maxsize
        maskj = np.expand_dims(maskj, axis=-1)

        box = np.array([0, 0, 0, 0])

        # Compute Bbx coordinates

        margin = 3
        xs = np.nonzero(np.sum(maskj, axis=0))[0]
        ys = np.nonzero(np.sum(maskj, axis=1))[0]
        box[1] = xs.min() - margin
        box[3] = xs.max() + margin
        box[0] = ys.min() - margin
        box[2] = ys.max() + margin

        if box[0] < 0: box[0] = 0
        if box[1] < 0: box[1] = 0

        h = box[2] - box[0]
        w = box[3] - box[1]
        if h < w:
            diff = w - h
            half = int(diff / 2)
            box[0] -= half
            if box[0] < 0:
                box[2] -= box[0]
                box[0] = 0
            else:
                box[2] += diff - half

            if box[2] > maskj.shape[0]:
                box[2] = maskj.shape[0]
        else:
            diff = h - w
            half = int(diff / 2)
            box[1] -= half
            if box[1] < 0:
                box[3] -= box[1]
                box[1] = 0
            else:
                box[3] += diff - half
            if box[3] > maskj.shape[1]:
                box[3] = maskj.shape[1]

        # if box[3] > resized_height: box[3] = resized_height - 1
        # if box[2] > resized_width: box[2] = resized_width - 1

        if box[3] == box[1]:
            box[3] += 1
        if box[0] == box[2]:
            box[2] += 1

        # bbx[box[0]:box[2], box[1]:box[3], :] = 1

        return box


if __name__ == '__main__':
    savepath = '/home/yam28/Documents/phdYoop/unsupervised-disentangling/datasets/giraffe_cs_fg/train_images'
    m_path = '/home/yam28/Documents/phdYoop/Stamps/dataset/train/giraffe'
    all_ms = sorted(glob.glob(os.path.join(m_path, '*.png')))
    df = getFGs(all_ms, '/home/yam28/Documents/phdYoop/datasets/COCO/train2017')
    for it, im in enumerate(df):
        name = os.path.basename(all_ms[it])
        cv2.imwrite(os.path.join(savepath,name), im)



    savepath = '/home/yam28/Documents/phdYoop/unsupervised-disentangling/datasets/giraffe_cs_fg/test_images'
    m_path = '/home/yam28/Documents/phdYoop/Stamps/dataset/val/giraffe'
    all_ms = sorted(glob.glob(os.path.join(m_path, '*.png')))
    df = getFGs(all_ms, '/home/yam28/Documents/phdYoop/datasets/COCO/val2017')
    for it, im in enumerate(df):
        name = os.path.basename(all_ms[it])
        cv2.imwrite(os.path.join(savepath,name), im)