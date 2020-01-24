import io
import os
import numpy as np
import h5py
import random

from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

object_names = ['airplane', 'bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'nightstand',
                'sofa', 'table', 'toilet', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'cone',
                'cup', 'curtain', 'door', 'flowerpot', 'glassbox', 'guitar', 'keyboard', 'lamp',
                'laptop', 'mantel', 'person', 'piano', 'plant', 'radio', 'rangehood', 'sink',
                'stairs', 'stool', 'tent', 'tvstand', 'vase', 'wardrobe', 'xbox']


object_names.sort()

def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]


def loadDataFile(filename, num_points):
    f = h5py.File(filename, "r")
    data = f['data'][:]
    label = f['label'][:]
    return data[:, 0:num_points, :], label

data_files = getDataFiles(list_filename="/home/yirus/Datasets/modelnet40_ply_hdf5_2048/train_files.txt")
total_files = len(data_files)
total_data = []
total_labels = []
for idx in range(total_files):
    current_data, current_label = loadDataFile(data_files[idx], num_points=1024)
    total_data.append(current_data)
    total_labels.append(current_label)

total_data = np.concatenate(total_data, axis=0)
total_labels = np.squeeze(np.concatenate(total_labels, axis=0))
total_data = total_data.astype(np.float32)
total_labels = total_labels.astype(np.int32)


shape_list = np.empty(40, dtype=np.object)
for i in range(shape_list.shape[0]):
    shape_list[i] = []
for i in range(total_labels.shape[0]):
    cls = total_labels[i]
    shape_list[cls].append(i)

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

# tot = 0
# for i in range(40):
#     tot += len(shape_list[i])
#     print("{} => {}".format(object_names[i], len(shape_list[i])))
# print("#total: {}".format(tot))

sample_ratio = 0.25

sample_index = []
sample_class_num = np.empty(40, dtype=np.int)

random.seed(0)
for i in range(40):
    np.random.shuffle(shape_list[i])
    sample_per_class = int(round(sample_ratio*len(shape_list[i])))
    select = shape_list[i][0:sample_per_class]
    # if i == 20:
    #     pts = total_data[select[0],:,:]
    #     im = pts2img(pts, 20)
    #     title = "pts_{}.png".format(object_names[i])
    #     im.save(os.path.join("/home/yirus/Datasets/modelnet40_ply_hdf5_2048", title))

    for j in select:
        sample_index.append(j)
    sample_class_num[i] = len(select)
    print("{} ===> {}".format(object_names[i], len(select)))

print("#sample: {}".format(len(sample_index)))
print("{}".format(sample_class_num.sum()))

fname = "gt_sample_{}.npy".format(len(sample_index))
np.save(fname, sample_index)
# shape = {}
# shape["bathtub"] = []
# shape["bed"] = []
# shape["chair"] = []
# shape["desk"] = []
# shape["dresser"] = []
# shape["monitor"] = []
# shape["nightstand"] = []
# shape["sofa"] = []
# shape["table"] = []
# shape["toilet"] = []
# shape["airplane"] = []
# shape["bench"] = []
# shape["bookshelf"] = []
# shape["bottle"] = []
# shape["bowl"] = []
# shape["car"] = []
# shape["cone"] = []
# shape["cup"] = []
# shape["curtain"] = []
# shape["door"] = []
# shape["flowerpot"] = []
# shape["glassbox"] = []
# shape["guitar"] = []
# shape["keyboard"] = []
# shape["lamp"] = []
# shape["laptop"] = []
# shape["mantel"] = []
# shape["person"] = []
# shape["piano"] = []
# shape["plant"] = []
# shape["radio"] = []
# shape["rangehood"] = []
# shape["sink"] = []
# shape["stairs"] = []
# shape["stool"] = []
# shape["tent"] = []
# shape["tvstand"] = []
# shape["vase"] = []
# shape["wardrobe"] = []
# shape["xbox"] = []








