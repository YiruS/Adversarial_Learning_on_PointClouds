import io
import random

import numpy as np
from PIL import Image
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

import torch
import torch.utils.data as data

class ModelNetDatasetGT(data.Dataset):
    def __init__(self,
                 root_list,
                 sample_list,
                 is_train,
                 npoints=1024,
                 data_augmentation=True):
        self.sample_list = sample_list
        self.npoints = npoints
        self.data_augmentation = data_augmentation
        self.data_files = self.getDataFiles(root_list)
        self.load_data()

        print("Loading GT data: {} ...".format(self.select_data.shape[0]))

        #if is_train:
        #    self.counts = self.__compute_class_probability()

    def getDataFiles(self, list_filename):
        return [line.rstrip() for line in open(list_filename)]

    def loadDataFile(self, filename, num_points):
        f = h5py.File(filename, "r")
        data = f['data'][:]
        label = f['label'][:]
        return data[:,0:num_points,:], label

    def load_data(self):
        total_files = len(self.data_files)
        total_data = []
        total_labels = []
        for idx in range(total_files):
            current_data, current_label = self.loadDataFile(self.data_files[idx], num_points=self.npoints)
            total_data.append(current_data)
            total_labels.append(current_label)

        total_data = np.concatenate(total_data, axis=0)
        total_labels = np.squeeze(np.concatenate(total_labels, axis=0))
        total_data = total_data.astype(np.float32)
        total_labels = total_labels.astype(np.int32)

        if isinstance(self.sample_list, np.ndarray):
            self.select_data = total_data[self.sample_list,:,:]
            self.select_labels = total_labels[self.sample_list]
        else:
            self.select_data = total_data.copy()
            self.select_labels = total_labels.copy()
        #print(self.total_data.shape, self.total_labels.shape)
        #print(np.min(self.total_data[:, 0]), np.min(self.total_data[:, 1]), np.min(self.total_data[:, 2]))
        #ßprint(np.max(self.total_data[:, 0]), np.max(self.total_data[:, 1]), np.max(self.total_data[:, 2]))

    def __getitem__(self, index):
        ptc, cls = self.select_data[index], self.select_labels[index]

        if self.data_augmentation:
            ptc = self.jitter_point_cloud(ptc)

        ptc, cls = ptc.astype(np.float32), cls.astype(np.int64)
        return ptc, cls

    def jitter_point_cloud(self, data, sigma=0.01, clip=0.05):
        """ Randomly jitter points. jittering is per point.
            Input:
              Nx3 array, original point clouds
            Return:
              Nx3 array, jittered point clouds
        """
        N, C = data.shape
        assert (clip > 0)
        jittered_data = np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
        jittered_data += data
        return jittered_data

    def __len__(self):
        return self.select_data.shape[0]

    def __compute_class_probability(self):
        counts = dict((i, 0) for i in range(40))
        sampleslist = np.arange(self.__len__())
        for i in sampleslist:
            pts, cls = self.__getitem__(i)
            if cls is not -1:
                for j in range(40):
                    counts[j] += np.sum(cls == j)
        return counts

    def get_class_probability(self):
        values = np.array(list(self.counts.values()))
        p_values = values / np.sum(values)
        return torch.Tensor(p_values)


class ModelNetDataset_noGT(data.Dataset):
    def __init__(self,
                 root_list,
                 sample_list,
                 npoints=1024,
                 data_augmentation=True):
        self.sample_list = sample_list
        self.npoints = npoints
        self.data_augmentation = data_augmentation
        self.data_files = self.getDataFiles(root_list)
        self.load_data()

        print("Loading NoGT data: {} ...".format(self.select_data.shape[0]))

    def getDataFiles(self, list_filename):
        return [line.rstrip() for line in open(list_filename)]

    def loadDataFile(self, filename, num_points):
        f = h5py.File(filename, "r")
        data = f['data'][:]
        label = f['label'][:]
        return data[:,0:num_points,:], label

    def load_data(self):
        total_files = len(self.data_files)
        total_data = []
        for idx in range(total_files):
            current_data, _ = self.loadDataFile(self.data_files[idx], num_points=self.npoints)
            total_data.append(current_data)

        total_data = np.concatenate(total_data, axis=0)
        total_data = total_data.astype(np.float32)

        self.select_data = total_data[self.sample_list,:,:]
        #print(self.total_data.shape, self.total_labels.shape)
        #print(np.min(self.total_data[:, 0]), np.min(self.total_data[:, 1]), np.min(self.total_data[:, 2]))
        #ßprint(np.max(self.total_data[:, 0]), np.max(self.total_data[:, 1]), np.max(self.total_data[:, 2]))

    def __getitem__(self, index):
        ptc = self.select_data[index]

        if self.data_augmentation:
            ptc = self.jitter_point_cloud(ptc)

        ptc = ptc.astype(np.float32)
        return ptc

    def jitter_point_cloud(self, data, sigma=0.01, clip=0.05):
        """ Randomly jitter points. jittering is per point.
            Input:
              Nx3 array, original point clouds
            Return:
              Nx3 array, jittered point clouds
        """
        N, C = data.shape
        assert (clip > 0)
        jittered_data = np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
        jittered_data += data
        return jittered_data

    def __len__(self):
        return self.select_data.shape[0]

def pts2img(pts, clr):
    def crop_image(image):
        image_data = np.asarray(image)
        assert (len(image_data.shape) == 3)
        image_data_bw = image_data.max(axis=2) if image_data.shape[-1] <= 3 else image_data[:, :, 3]
        non_empty_columns = np.where(image_data_bw.max(axis=0) > 0)[0]
        non_empty_rows = np.where(image_data_bw.max(axis=1) > 0)[0]
        cropBox = (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))
        image_data_new = image_data[cropBox[0]:cropBox[1] + 1, cropBox[2]:cropBox[3] + 1, :]
        return Image.fromarray(image_data_new)
    plt.close('all')
    fig = plt.figure()
    fig.set_rasterized(True)
    ax = axes3d.Axes3D(fig)
    pts -= np.mean(pts, axis=0) #demean

    ax.view_init(20,200) # (20,30) for motor, (20, 20) for car, (20,210), (180, 60) for lamp, ax.view_init(20,110) for general, (180,60) for lamp, (20, 200) for mug and pistol
    ax.set_alpha(255)
    ax.set_aspect('auto')
    min_lim = pts.min()
    max_lim = pts.max()
    ax.set_xlim3d(min_lim,max_lim)
    ax.set_ylim3d(min_lim,max_lim)
    ax.set_zlim3d(min_lim,max_lim)

    ax.scatter(
        pts[:, 0], pts[:, 1], pts[:, 2],
        zdir='x',
        s=20,
        edgecolors=(0.5, 0.5, 0.5)  # (0.5,0.5,0.5)
    )

    ax.set_axis_off()
    ax.set_facecolor((1,1,1,0))
    ax.set_rasterized(True)
    ax.set_rasterization_zorder(1)
    ax.set_facecolor("white")
    buf = io.BytesIO()
    plt.savefig(
        buf, format='png', transparent=True,
        bbox_inches='tight', pad_inches=0,
        rasterized=True,
        dpi=200
    )
    buf.seek(0)
    im = Image.open(buf)
    im = crop_image(im)
    buf.close()
    return im

if __name__ == '__main__':
    dataset = ModelNetDataset(root_list="/home/yirus/Datasets/modelnet40_ply_hdf5_2048/train_files.txt")
    print("#train: {}".format(len(dataset)))
    dataloader = data.DataLoader(dataset, batch_size=1)
    # for i, data in enumerate(dataloader):
    #     pts, cls = data
    #     pts, cls = pts.cpu().numpy().squeeze(), cls.cpu().numpy().squeeze()
    #     if i<10:
    #         im = pts2img(pts, int(cls))
    #         title = "pts_{}.png".format(i)
    #         im.save(os.path.join("/home/yirus/Datasets/modelnet40_ply_hdf5_2048", title))
    # ps, cls, seg = data[0]
    # print("#pts: {}, type: {}".format(ps.size(), ps.type()))
    # print("class: {}, type: {}".format(cls.size(), cls.type()))
    # print("seg: {}, type: {}".format(seg.size(), seg.type()))

    shapenet = ModelNetDataset(root_list="/home/yirus/Datasets/modelnet40_ply_hdf5_2048/test_files.txt")
    print("#test: {}".format(len(shapenet)))
