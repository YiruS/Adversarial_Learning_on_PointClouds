
import numpy as np

object_names = ['Airplane', 'Bag', 'Cap', 'Car', 'Chair', 'Earphone', 'Guitar', 'Knife', 'Lamp',
                'Laptop', 'Motorbike', 'Mug', 'Pistol', 'Rocket', 'Skateboard', 'Table']
seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43], 'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
part_colors = [[0.65, 0.95, 0.05], [0.35, 0.05, 0.35], [0.65, 0.35, 0.65], [0.95, 0.95, 0.65], [0.95, 0.65, 0.05], [0.35, 0.05, 0.05], [0.65, 0.05, 0.05], [0.65, 0.35, 0.95], [0.05, 0.05, 0.65], [0.65, 0.05, 0.35], [0.05, 0.35, 0.35], [0.65, 0.65, 0.35], [0.35, 0.95, 0.05], [0.05, 0.35, 0.65], [0.95, 0.95, 0.35], [0.65, 0.65, 0.65], [0.95, 0.95, 0.05], [0.65, 0.35, 0.05], [0.35, 0.65, 0.05], [0.95, 0.65, 0.95], [0.95, 0.35, 0.65], [0.05, 0.65, 0.95], [0.65, 0.95, 0.65], [0.95, 0.35, 0.95], [0.05, 0.05, 0.95], [0.65, 0.05, 0.95], [0.65, 0.05, 0.65], [0.35, 0.35, 0.95], [0.95, 0.95, 0.95], [0.05, 0.05, 0.05], [0.05, 0.35, 0.95], [0.65, 0.95, 0.95], [0.95, 0.05, 0.05], [0.35, 0.95, 0.35], [0.05, 0.35, 0.05], [0.05, 0.65, 0.35], [0.05, 0.95, 0.05], [0.95, 0.65, 0.65], [0.35, 0.95, 0.95], [0.05, 0.95, 0.35], [0.95, 0.35, 0.05], [0.65, 0.35, 0.35], [0.35, 0.95, 0.65], [0.35, 0.35, 0.65], [0.65, 0.95, 0.35], [0.05, 0.95, 0.65], [0.65, 0.65, 0.95], [0.35, 0.05, 0.95], [0.35, 0.65, 0.95], [0.35, 0.05, 0.65]]

cls_shape_cnt = {}
for shape_cls in object_names:
    cls_shape_cnt[shape_cls] = 0

seg_label_to_cat = {} # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat

shape_ious = {cat: [] for cat in seg_classes.keys()}

def get_iou(gt, pred, cls_gt):
    cls_name = object_names[cls_gt]
    shape_cat = seg_classes[cls_name]
    part_ious = [0.0 for _ in range(len(shape_cat))]
    for l in shape_cat:
        if (np.sum(pred == l) == 0) and (np.sum(gt == l) == 0):  # part is not present, no prediction as well
            part_ious[l - shape_cat[0]] = 1.0
        else:
            part_ious[l - shape_cat[0]] = np.sum((gt == l) & (pred == l)) / float(
                np.sum((gt == l) | (pred == l)))
    shape_ious[cls_name].append(np.mean(part_ious))
    return np.mean(part_ious)

def batch_get_iou(batch_pred, batch_seg, batch_cls):
    batch_iou = []
    for i in range(batch_pred.shape[0]):
        cls = np.argmax(batch_cls[i,:])
        iou = get_iou(batch_seg[i], batch_pred[i],cls)
        batch_iou.append(iou)
    return batch_iou
