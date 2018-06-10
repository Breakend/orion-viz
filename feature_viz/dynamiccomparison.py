#!/usr/bin/env python

""" A script for plotting feature based analysis of hyperparameter optimization performance
"""

from __future__ import print_function
import os
import sys
import argparse
import numpy as np
import pickle
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
import datetime
from scipy import stats
import uuid
from mpl_toolkits.mplot3d import Axes3D

# plt.rcParams['text.usetex'] = True
from matplotlib import rcParams
rcParams.update({
    'figure.autolayout': True,
    'axes.labelsize': 'xx-large',
    'axes.titlesize' : 'xx-large',
    'xtick.labelsize': 'x-large',
    'ytick.labelsize': 'x-large',
    'legend.fontsize' : 'small',
    'font.size': 18,
    'font.family': 'Times New Roman',
    'font.serif': 'Times'
    })
def _get_feature_from_list(l, feature):
    for x in l:
        if x['name'] == feature:
            return x['value']

    raise ValueError("Could not find feature with name {} in data list: {}.".format(feature, str(l)))

def _get_matrix_from_features(data, features, n_target_components):
    # TODO:
    all_data = []
    for data_point in data:
        concated = np.array([_get_feature_from_list(data_point['params'],feature) for feature in features])
        all_data.append(concated)

    X = np.vstack(all_data)
    projected = False
    if n_target_components < len(features):
        pca = PCA(n_components=n_target_components)
        X = pca.fit(X).transform(X)
        projected = True

    return X, projected

def get_lines(data, evalmetric):
    bigxs = []
    bigys = []
    for datapoint in data:
        xs = []
        ys = []
        try:
            for x in sorted(datapoint['dynamic'], key=lambda y: _get_feature_from_list(y['params'], 'steps')):
                steps =  _get_feature_from_list(x['params'], 'steps')
                met = _get_feature_from_list(x['results'], evalmetric)
                xs.append(steps)
                ys.append(met)
        except:
            print("problem with datapoint")

        bigxs.append(np.array(xs))
        bigys.append(np.array(ys))

    return bigxs, bigys



from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.text import Annotation

class Annotation3D(Annotation):
    '''Annotate the point xyz with text s

    From: https://stackoverflow.com/questions/10374930/matplotlib-annotating-a-3d-scatter-plot
    '''

    def __init__(self, s, xyz, *args, **kwargs):
        Annotation.__init__(self,s, xy=(0,0), *args, **kwargs)
        self._verts3d = xyz

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.xy=(xs,ys)
        Annotation.draw(self, renderer)

def annotate3D(ax, s, *args, **kwargs):
    '''add anotation text s to to Axes3d ax'''

    tag = Annotation3D(s, *args, **kwargs)
    ax.add_artist(tag)

def _get_results_with_name(data, name):
    # TODO:

    all_data = []
    for data_point in data:
        concated = np.array([_get_feature_from_list(data_point['results'], name)])
        all_data.append(concated)

    Y = np.vstack(all_data)

    return Y

def _null_aware_key_comparator(x):

    if x["start_time"] is not None:
        return x["start_time"]
    else:
        return 1

def main(arguments):

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('data', nargs="+", type=argparse.FileType('rb'), help="Input file as a pickle in Orion format.")
    parser.add_argument('--loss_interpolation_type', choices=["colour", "none"], help="How to show the validation loss value if at all.", default="colour")
    parser.add_argument('--evaluation_metric', help="The evaluation metric to use for showing performance of features.")
    parser.add_argument('--title', default=None, help="How would you like to title your plot?")
    parser.add_argument('--save', action="store_true", default=False, help="Save the plot?")
    parser.add_argument('--fig_size', default=[16,8], nargs="+", help="The size of the figure", type=int)
    parser.add_argument('--best_percent_filter', type=float, default=100.0, help="Take only the top 25% of samples according to the evaluation_metric. If negative takes the bottom percent.")
    parser.add_argument('--lower_better', action="store_true", default=False, help="Inverse the colour if lower is better for the evaluation_metric")
    parser.add_argument('--names', nargs="+", type=str, help="names of data")
    parser.add_argument('--ymax', default=None, type=float)
    args = parser.parse_args(arguments)
    print(args)
    fig = plt.figure(figsize=args.fig_size)

    ax = fig.gca()

    for data_file, name in zip(args.data, args.names):
        data = pickle.load(data_file, encoding='bytes')
        data2 = sorted(data, key=_null_aware_key_comparator)
        data = data2


        X, Y = get_lines(data, args.evaluation_metric)

        Yoverall = _get_results_with_name(data, args.evaluation_metric)
        filtered = []
        if args.evaluation_metric:
            if args.best_percent_filter < 100.0:
                percentile = np.percentile(Yoverall, np.abs(args.best_percent_filter))
                if args.best_percent_filter < 0:
                    filtered = Yoverall[:,0]<=percentile
                else:
                    filtered = Yoverall[:,0]>=percentile

        for i, (xline, yline) in enumerate(zip(X,Y)):
            if len(filtered) > 0 and filtered[i]:
                print("Skipping Trial {} outside of percentile".format(i))
                continue
            plt.plot(xline, yline, label="{} (Trial {})".format(name, i), linewidth=2)

    axis_font = {'fontname':'Arial', 'size':'32'}

    plt.xlabel("Steps", **axis_font)
    plt.ylabel(args.evaluation_metric, **axis_font)
    if args.ymax:
        plt.ylim(ymax=args.ymax)

    plt.legend()

    if args.title:
        plt.title(args.title, **axis_font)

    if args.save:
        unique_filename = str(uuid.uuid4())
        fig.savefig("{}.pdf".format(unique_filename), dpi=fig.dpi, bbox_inches='tight')
        print("Saved to: {}.pdf".format(unique_filename))
    else:
        plt.show()


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
