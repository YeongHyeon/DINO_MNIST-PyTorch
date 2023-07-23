import os, glob, shutil
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def make_dir(path, refresh=False):

    try: os.mkdir(path)
    except:
        if(refresh):
            shutil.rmtree(path)
            os.mkdir(path)

def sorted_list(path):

    tmplist = glob.glob(path)
    tmplist.sort()

    return tmplist

def min_max_norm(x):

    return (x - x.min() + (1e-30)) / (x.max() - x.min() + (1e-30))

def nhwc2nchw(x):

    return np.transpose(x, [0, 3, 1, 2])

def nchw2nhwc(x):

    return np.transpose(x, [0, 2, 3, 1])

def discrete_cmap(N, base_cmap=None):

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)

    return base.from_list(cmap_name, color_list, N)

def plot_projection(emb, y, sample=-1, mode='pca', savepath=None):

    if(mode == 'pca'):
        projector = PCA(n_components=2)
    elif(mode == 'tsne'):
        projector = TSNE(n_components=2)
    
    projected = projector.fit_transform(emb)

    if((sample == -1) or (y.shape[0] < sample)):
        prj_x, prj_y = projected[:, 0], projected[:, 1]
    else:
        step = projected.shape[0] // sample
        prj_x, prj_y = projected[::step, 0], projected[::step, 1]
        y = y[::step]
    
    plt.figure(figsize=(6, 6), dpi=200)
    # plt.scatter(prj_x, prj_y, c=y, \
    #     marker='o', edgecolor='none', cmap=discrete_cmap(10, 'jet'))
    for idx_y in range(10):
        plt.scatter(prj_x[y==idx_y], prj_y[y==idx_y], \
            marker='o', edgecolor='none', label='%d' %(idx_y))
    plt.legend(loc='upper right')
    plt.grid()
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()