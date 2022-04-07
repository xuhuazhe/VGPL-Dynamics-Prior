from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
import scipy
from scipy import optimize

import matplotlib

matplotlib.rcParams["legend.loc"] = 'lower right'

def train_plot_curves(iters, loss, path=''):
    plt.figure(figsize=[16,9])
    plt.plot(iters, loss)
    plt.xlabel('iterations', fontsize=30)
    plt.ylabel('loss', fontsize=30)
    plt.title('Training Loss', fontsize=35)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)

    if len(path) > 0:
        plt.savefig(path)
    else:
        plt.show()

def eval_plot_curves_with_bar(loss_mean, loss_std, colors=['orange', 'royalblue'], 
    alpha_fill=0.3, ax=None, path=''):
    iters, loss_mean_emd, loss_mean_chamfer, loss_mean_iou = loss_mean.T
    _, loss_std_emd, loss_std_chamfer, loss_std_iou = loss_std.T
    plt.figure(figsize=[16, 9])

    emd_min = loss_mean_emd - loss_std_emd
    emd_max = loss_mean_emd + loss_std_emd

    chamfer_min = loss_mean_chamfer - loss_std_chamfer
    chamfer_max = loss_mean_chamfer + loss_std_chamfer

    plt.plot(iters, loss_mean_emd, color=colors[0], linewidth=6, label='EMD')
    plt.fill_between(iters, emd_max, emd_min, color=colors[0], alpha=alpha_fill)

    plt.plot(iters, loss_mean_chamfer, color=colors[1], linewidth=6, label='Chamfer')
    plt.fill_between(iters, chamfer_max, chamfer_min, color=colors[1], alpha=alpha_fill)

    plt.xlabel('Time Steps', fontsize=30)
    plt.ylabel('Loss', fontsize=30)
    plt.title('Dyanmics Model Evaluation Loss', fontsize=35)
    plt.legend(fontsize=30)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)


    if len(path) > 0:
        plt.savefig(path)
    else:
        plt.show()

def eval_plot_curves_with_bar_iou(loss_mean, loss_std, colors=['orange'], 
    alpha_fill=0.3, ax=None, path=''):
    iters, loss_mean_emd, loss_mean_chamfer, loss_mean_iou = loss_mean.T
    _, loss_std_emd, loss_std_chamfer, loss_std_iou = loss_std.T
    plt.figure(figsize=[16, 9])

    iou_min = loss_mean_iou - loss_std_iou
    iou_max = loss_mean_iou + loss_std_iou

    plt.plot(iters, loss_mean_iou, color=colors[0], linewidth=6, label='IOU')
    plt.fill_between(iters, iou_max, iou_min, color=colors[0], alpha=alpha_fill)

    plt.xlabel('Time Steps', fontsize=30)
    plt.ylabel('Loss', fontsize=30)
    plt.title('Dyanmics Model Evaluation Loss', fontsize=35)
    plt.legend(fontsize=30)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)

    if len(path) > 0:
        plt.savefig(path)
    else:
        plt.show()

def eval_plot_curves(loss_list, path=''):
    iters, loss_emd, loss_chamfer, loss_uh = map(list, zip(*loss_list))
    plt.figure(figsize=[16, 9])
    plt.plot(iters, loss_emd, linewidth=6, label='EMD')
    plt.plot(iters, loss_chamfer, linewidth=6, label='Chamfer')
    plt.plot(iters, loss_uh, linewidth=6, color='r', label='Hausdorff')
    plt.xlabel('frames', fontsize=30)
    plt.ylabel('loss', fontsize=30)
    plt.title('Test Loss', fontsize=35)
    plt.legend(fontsize=30)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)

    if len(path) > 0:
        plt.savefig(path)
    else:
        plt.show()

def em_distance(self, x, y):
    x_ = x[:, :, None, :].repeat(1, 1, y.size(1), 1)  # x: [B, N, M, D]
    y_ = y[:, None, :, :].repeat(1, x.size(1), 1, 1)  # y: [B, N, M, D]
    dis = torch.norm(torch.add(x_, -y_), 2, dim=3)  # dis: [B, N, M]
    x_list = []
    y_list = []
    # x.requires_grad = True
    # y.requires_grad = True
    for i in range(dis.shape[0]):
        cost_matrix = dis[i].detach().cpu().numpy()
        try:
            ind1, ind2 = scipy.optimize.linear_sum_assignment(cost_matrix, maximize=False)
        except:
            import pdb;
            pdb.set_trace()
        x_list.append(x[i, ind1])
        y_list.append(y[i, ind2])
        # x[i] = x[i, ind1]
        # y[i] = y[i, ind2]
    new_x = torch.stack(x_list)
    new_y = torch.stack(y_list)
    emd = torch.mean(torch.norm(torch.add(new_x, -new_y), 2, dim=2))
    return emd

def matched_motion(p_cur, p_prev, n_particles):
    x = p_cur[:, :n_particles, :]
    y = p_prev[:, :n_particles, :]

    x_ = x[:, :, None, :].repeat(1, 1, y.size(1), 1)
    y_ = y[:, None, :, :].repeat(1, y.size(1), 1, 1)
    dis = torch.norm(torch.add(x_, -y_), 2, dim=3)
    x_list = []
    y_list = []
    for i in range(dis.shape[0]):
        cost_matrix = dis[i].detach().cpu().numpy()
        ind1, ind2 = scipy.optimize.linear_sum_assignment(cost_matrix, maximize=False)
        x_list.append(x[i, ind1])
        y_list.append(y[i, ind2])
    new_x = torch.stack(x_list)
    new_y = torch.stack(y_list)
    p_cur_new = torch.cat((new_x, p_cur[:, n_particles:, :]), dim=1)
    p_prev_new = torch.cat((new_y, p_prev[:, n_particles:, :]), dim=1)
    dist = torch.add(p_cur_new, -p_prev_new)
    return dist


def my_collate(batch):
    len_batch = len(batch[0])
    len_rel = 3

    ret = []
    for i in range(len_batch - len_rel - 1):
        d = [item[i] for item in batch]
        if isinstance(d[0], int):
            d = torch.LongTensor(d)
        else:
            d = torch.FloatTensor(torch.stack(d))
        ret.append(d)

    # processing relations
    # R: B x seq_length x n_rel x (n_p + n_s)
    for i in range(len_rel):
        R = [item[-len_rel + i - 1] for item in batch]
        max_n_rel = 0
        seq_length, _, N = R[0].size()
        for j in range(len(R)):
            max_n_rel = max(max_n_rel, R[j].size(1))
        for j in range(len(R)):
            r = R[j]
            r = torch.cat([r, torch.zeros(seq_length, max_n_rel - r.size(1), N)], 1)
            R[j] = r

        R = torch.FloatTensor(torch.stack(R))

        ret.append(R)

    # std reg
    d = [item[-1] for item in batch]
    if d[0] is not None:
        if isinstance(d[0], int):
            d = torch.LongTensor(d)
        else:
            d = torch.FloatTensor(torch.stack(d))
        ret.append(d)
    else:
        ret.append(None)
    return tuple(ret)


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()

    def close(self):
        self.__del__()


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def check_gradient(step):
    def hook(grad):
        print(step, torch.mean(grad, 1)[:4])
    return hook


def add_log(fn, content, is_append=True):
    if is_append:
        with open(fn, "a+") as f:
            f.write(content)
    else:
        with open(fn, "w+") as f:
            f.write(content)


def rand_int(lo, hi):
    return np.random.randint(lo, hi)


def rand_float(lo, hi):
    return np.random.rand() * (hi - lo) + lo


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_var(tensor, use_gpu, requires_grad=False):
    if use_gpu:
        return Variable(torch.FloatTensor(tensor).cuda(),
                        requires_grad=requires_grad)
    else:
        return Variable(torch.FloatTensor(tensor),
                        requires_grad=requires_grad)


def make_graph(log, title, args):
    """make a loss graph"""
    plt.plot(log)
    plt.xlabel('iter')
    plt.ylabel('loss')

    title + '_loss_graph'
    plt.title(title)
    plt.savefig(os.path.join(args.logf, title + '.png'))
    plt.close()


def get_color_from_prob(prob, colors):
    # there's only one instance
    if len(colors) == 1:
        return colors[0] * prob
    elif len(prob) == 1:
        return colors * prob[0]
    else:
        res = np.zeros(4)
        for i in range(len(prob)):
            res += prob[i] * colors[i]
        return res


def create_instance_colors(n):
    # TODO: come up with a better way to initialize instance colors
    return np.array([
        [1., 0., 0., 1.],
        [0., 1., 0., 1.],
        [0., 0., 1., 1.],
        [1., 1., 0., 1.],
        [1., 0., 1., 1.]])[:n]


def convert_groups_to_colors(group, n_particles, n_rigid_instances, instance_colors, env=None):
    """
    Convert grouping to RGB colors of shape (n_particles, 4)
    :param grouping: [p_rigid, p_instance, physics_param]
    :return: RGB values that can be set as color densities
    """
    # p_rigid: n_instance
    # p_instance: n_p x n_instance
    p_rigid, p_instance = group[:2]

    p = p_instance

    colors = np.empty((n_particles, 4))

    for i in range(n_particles):
        colors[i] = get_color_from_prob(p[i], instance_colors)

    # print("colors", colors)
    return colors


def visualize_point_clouds(point_clouds, c=['b', 'r'], view=None, store=False, store_path=''):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')

    frame = plt.gca()
    frame.axes.xaxis.set_ticklabels([])
    frame.axes.yaxis.set_ticklabels([])
    frame.axes.zaxis.set_ticklabels([])

    for i in range(len(point_clouds)):
        points = point_clouds[i]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=c[i], s=10, alpha=0.3)

    X, Y, Z = point_clouds[0][:, 0], point_clouds[0][:, 1], point_clouds[0][:, 2]

    max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max()
    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (X.max() + X.min())
    Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (Y.max() + Y.min())
    Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (Z.max() + Z.min())
    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')

    ax.grid(False)
    plt.show()

    if view is None:
        view = 0, 0
    ax.view_init(view[0], view[1])
    plt.draw()

    # plt.pause(5)

    if store:
        os.system('mkdir -p ' + store_path)
        fig.savefig(os.path.join(store_path, "vis.png"), bbox_inches='tight')

    '''
    for angle in range(0, 360, 2):
        ax.view_init(90, angle)
        plt.draw()
        # plt.pause(.001)

        if store:
            if angle % 100 == 0:
                print("Saving frame %d / %d" % (angle, 360))

            os.system('mkdir -p ' + store_path)
            fig.savefig(os.path.join(store_path, "%d.png" % angle), bbox_inches='tight')
    '''


def quatFromAxisAngle(axis, angle):
    axis /= np.linalg.norm(axis)

    half = angle * 0.5
    w = np.cos(half)

    sin_theta_over_two = np.sin(half)
    axis *= sin_theta_over_two

    quat = np.array([axis[0], axis[1], axis[2], w])

    return quat


def quatFromAxisAngle_var(axis, angle):
    axis /= torch.norm(axis)

    half = angle * 0.5
    w = torch.cos(half)

    sin_theta_over_two = torch.sin(half)
    axis *= sin_theta_over_two

    quat = torch.cat([axis, w])
    # print("quat size", quat.size())

    return quat


class ChamferLoss(torch.nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()

    def chamfer_distance(self, x, y):
        # x: [N, D]
        # y: [M, D]
        x = x.repeat(y.size(0), 1, 1)  # x: [M, N, D]
        x = x.transpose(0, 1)  # x: [N, M, D]
        y = y.repeat(x.size(0), 1, 1)  # y: [N, M, D]
        dis = torch.norm(torch.add(x, -y), 2, dim=2)  # dis: [N, M]
        dis_xy = torch.mean(torch.min(dis, dim=1)[0])  # dis_xy: mean over N
        dis_yx = torch.mean(torch.min(dis, dim=0)[0])  # dis_yx: mean over M

        return dis_xy + dis_yx

    def __call__(self, pred, label):
        return self.chamfer_distance(pred, label)


def get_l2_loss(g):
    num_particles = len(g)
    return torch.norm(num_particles - torch.norm(g, dim=1, keepdim=True))
