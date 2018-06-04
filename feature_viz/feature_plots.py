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
    parser.add_argument('data', type=argparse.FileType('rb'), help="Input file as a pickle in Orion format.")
    parser.add_argument('--features', nargs="+", type=str, default=None, help="An optional list of features to use rather than the entire list.")
    parser.add_argument('--projection_type', choices=['2d', '3d'], default='2d', help="The type of projection, a 2D plane, a 3D plane, or a heatmap.")
    parser.add_argument('--loss_interpolation_type', choices=["colour", "heatmap", "none"], help="How to show the validation loss value if at all.", default="colour")
    parser.add_argument("--points", choices=["o"], default="o", help="How to draw the individual data points.")
    parser.add_argument("--point_label", choices=["order", "none"], help="If adding text annotation to the points in the scatter plot, what to annotate. Order is the order in which the hyperparameters were tried by start_time.")
    parser.add_argument('--evaluation_metric', help="The evaluation metric to use for showing performance of features.")
    parser.add_argument('--title', default=None, help="How would you like to title your plot?")
    parser.add_argument('--save', action="store_true", default=False, help="Save the plot?")
    parser.add_argument('--use_time', action="store_true", help="Makes time the X axis to see progression across state space.")
    parser.add_argument('--time_axis', default="X", choices=["X", "colour","X+colour"], help="By default if use_time argument is enabled, plots as X axis being time. If -1, sets color scheme to be represented by start_time instead of loss.")
    parser.add_argument('--fig_size', default=[16,8], nargs="+", help="The size of the figure", type=int)
    parser.add_argument('--best_percent_filter', type=float, default=100.0, help="Take only the top 25% of samples according to the evaluation_metric. If negative takes the bottom percent.")
    parser.add_argument('--lower_better', action="store_true", default=False, help="Inverse the colour if lower is better for the evaluation_metric")
    parser.add_argument('--bins', help="The number of bins for a heatmap projection to use.", default=10, type=int)
    parser.add_argument('--heatmap_interpolate_missing', action="store_true", default=False, help="If the heatmap doesn't have enough data, colour the missing bins according to the worst value.")
    parser.add_argument('--time_rank', action="store_true", default=False, help="Uses rank when visualizing or labeling time instead of timestamp.")
    args = parser.parse_args(arguments)
    print(args)

    data = pickle.load(args.data, encoding='bytes')
    data2 = sorted(data, key=_null_aware_key_comparator)
    data = data2

    # currently only support real
    supported_feature_types = ['real']

    if not args.features:
        features = [x['name'] for x in data[0]["params"] if x['type'] in supported_feature_types]
        ignored_features = [x['name'] for x in data[0]["params"] if x['type'] not in supported_feature_types]
        print('Ignoring features {} because currently not supported. Only support {}.'.format(ignored_features, supported_feature_types))
    else:
        features = args.features

    print("Using features {}".format(features))

    if args.use_time and args.time_axis != "colour":
        required_features = { "3d" : 2, "2d" : 1}
    else:
        required_features = { "3d" : 3, "2d" : 2, "heatmap": 2}

    if len(features) < required_features[args.projection_type]:
        raise ValueError("To make a graph need at least {} features, have {}.".format(required_features[args.projection_type], len(features)))


    X, projected = _get_matrix_from_features(data, features, required_features[args.projection_type])

    if args.use_time:
        if args.time_axis != "colour":
            X = np.concatenate([np.arange(X.shape[0]).reshape(-1,1),X], axis=1)

    fig = plt.figure(figsize=args.fig_size)
    if args.projection_type == "3d":
        ax = Axes3D(fig)
    else:
        ax = fig.gca()

    # TODO: handle non-numeric features
    if args.evaluation_metric:
        Y = _get_results_with_name(data, args.evaluation_metric)
        if args.best_percent_filter < 100.0:
            percentile = np.percentile(Y, np.abs(args.best_percent_filter))
            if args.best_percent_filter < 0:
                X2 = X[Y[:,0]<=percentile]
                Y2 = Y[Y[:,0]<=percentile]
                X = X2
                Y = Y2
            else:
                X2 = X[Y[:,0]>=percentile]
                Y2 = Y[Y[:,0]>=percentile]
                X = X2
                Y = Y2

    if args.use_time and (args.time_axis == "colour" or args.time_axis == "X+colour"):
        # Sometimes we want to colour things by when they were created
        Ytime = np.array([(x["start_time"]-datetime.datetime(1970,1,1)).total_seconds() for x in data])

        if args.time_rank:
            order = Ytime.reshape(-1).argsort()
            ranks = order.argsort()
            Ytime = ranks.reshape(-1,1)

        # TODO: make this less repeated code
        if args.best_percent_filter < 100.0 and args.evaluation_metric:
            Y = _get_results_with_name(data, args.evaluation_metric)
            if args.best_percent_filter < 100.0:
                percentile = np.percentile(Y, np.abs(args.best_percent_filter))
                if args.best_percent_filter < 0:
                    Ytime = Ytime[Y[:,0]>=percentile]
                else:
                    Ytime = Ytime[Y[:,0]<=percentile]
        Y = Ytime

    if args.loss_interpolation_type == "colour":
        if args.lower_better:
            cm = plt.cm.get_cmap('RdYlBu')
        else:
            cm = plt.cm.get_cmap('RdYlBu_r')
        cmap=cm
        normalize = matplotlib.colors.Normalize(vmin=np.min(Y), vmax=np.max(Y))
        if args.projection_type == "3d":
            sc = ax.scatter(X[:,0], X[:, 1], X[:, 2], s=50, c=Y.reshape(-1), alpha=.7, norm=normalize, lw = 0, cmap=cm)

        else:
            sc = plt.scatter(X[:,0], X[:, 1], c=Y.reshape(-1), s=50, alpha=.7, norm=normalize, lw = 0, cmap=cm)

        clbar = plt.colorbar(sc)

        if args.use_time and (args.time_axis == "colour" or args.time_axis == "X+colour"):
            if not args.time_rank:
                clbar.set_label("Start Time")
            else:
                clbar.set_label("Trials")
        else:
            clbar.set_label(args.evaluation_metric)

    elif args.loss_interpolation_type == "heatmap":
        if args.projection_type == "3d":
            raise NotImplementedError("3d projection not yet supported as Heatmap")

        # plt.hexbin(x,y)
        H, xedges, yedges, binnumber = stats.binned_statistic_2d(X[:,0], X[:,1], Y.reshape(-1), statistic='mean', bins=(args.bins, args.bins))
        # plt.clf()
        H = np.ma.masked_invalid(H)
        if args.heatmap_interpolate_missing:
            if args.lower_better:
                H[np.isnan(H)] = np.max(Y)
            else:
                H[np.isnan(H)] = np.min(Y)


        xi, yi = np.meshgrid(xedges, yedges)
        cmap  = 'RdYlBu_r' if not args.lower_better else 'RdYlBu'
        sc = plt.pcolormesh(xi, yi, H.T, cmap=cmap)
        clbar = plt.colorbar(sc)
        plt.xlim(xmin=np.min(xi), xmax=np.max(xi))
        plt.ylim(ymin=np.min(yi), ymax=np.max(yi))
        clbar.set_label(args.evaluation_metric)
        # plt.imshow(heatmap)
    elif args.loss_interpolation_type == "none":
        if args.projection_type == "3d":
            sc = ax.scatter(X[:,0], X[:, 1], X[:, 2], s=50, color="teal", alpha=.5, lw = 0)

        else:
            sc = plt.scatter(X[:,0], X[:, 1], color="teal", s=50, alpha=.5, lw = 0)

    if args.use_time:
        if len(features) == required_features[args.projection_type]:
            if args.projection_type == "2d":
                if args.time_axis == "colour":
                    plt.xlabel(features[0])
                    plt.ylabel(features[1])
                else:
                    plt.xlabel('Trials')
                    plt.ylabel(features[0])
            else:
                if args.time_axis == "colour":
                    ax.set_ylabel(features[0])
                    ax.set_zlabel(features[1])
                    ax.set_zlabel(features[2])
                else:
                    ax.set_xlabel('Trials')
                    ax.set_ylabel(features[0])
                    ax.set_zlabel(features[1])
        else:
            if args.projection_type == "2d":
                plt.xlabel("Trials")
                plt.ylabel("PCA Component (Features {})".format(", ".join(features)))
            else:
                if args.time_axis == "colour":
                    ax.set_xlabel("PCA Component 1 (Features {})".format(", ".join(features)))
                    ax.set_ylabel("PCA Component 2 (Features {})".format(", ".join(features)))
                    ax.set_zlabel("PCA Component 3 (Features {})".format(", ".join(features)))
                else:
                    ax.set_xlabel("Trials")
                    ax.set_ylabel("PCA Component 1 (Features {})".format(", ".join(features)))
                    ax.set_zlabel("PCA Component 2 (Features {})".format(", ".join(features)))
    else:
        if len(features) == required_features[args.projection_type]:
            # TODO: handle 3d projection
            if args.projection_type == "2d":
                plt.xlabel(features[0])
                plt.ylabel(features[1])
            else:
                ax.set_xlabel(features[0])
                ax.set_ylabel(features[1])
                ax.set_zlabel(features[2])
        else:
            if args.projection_type == "2d":
                plt.xlabel("PCA Component 1 (Features {})".format(", ".join(features)))
                plt.ylabel("PCA Component 2 (Features {})".format(", ".join(features)))
            else:
                ax.set_xlabel("PCA Component 1 (Features {})".format(", ".join(features)))
                ax.set_ylabel("PCA Component 2 (Features {})".format(", ".join(features)))
                ax.set_zlabel("PCA Component 3 (Features {})".format(", ".join(features)))
    if args.title:
        plt.title(args.title)

    if args.point_label is not "none":

        if args.point_label == "order":
            for i, txt in enumerate(X):
                if args.projection_type == "3d":
                    raise NotImplementedError("3D projection with point labels is not currently supported")
                    # TODO: for some reason this doesn't work, need to debug
                    annotate3D(ax, str(i), (X[i,0],X[i,1], X[i,2]))
                else:
                    ax.annotate(str(i), (X[i,0],X[i,1]))

    if args.save:
        unique_filename = str(uuid.uuid4())
        fig.savefig("{}.pdf".format(unique_filename), dpi=fig.dpi, bbox_inches='tight')
        print("Saved to: {}.pdf".format(unique_filename))
    else:
        plt.show()


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

# 2. specify size of points based on loss
# ~30min
# 3. heatmap
# ~30 min
# 4. create order labels on heatmap
# ~30 min
# 4. script to plot loss against hyperparameter or loss based on resource time with each line being a hyperparameter configuration
   # ~1hr
# 5. make tree thing
# convert pbt data to orion format and infer parents by closest possible parent where possible parents are determined by timestamp
# 2-3hrs

#
# Script 1
#
# Arguments:
#     features : features to use
#     pca : if >2 features can project them into 2d plane
#     dimensions: 2D or 3d projection
#     loss_interpolation : ["colour", "pointsize" , None] depending how want to indicate better or worse loss
#     interpolate_phylo : interpolate a phylogentic connection if one is not provided (TODO: might be good to get data points to include ancestor references)
#
#     # possibly just move to another script
#     heatmap: draw a heatmap instead of points
#     heatmap_points: include point labels with heatmap ["order", "x", None]
#
#
# Script 2
#
# Dynamic resource, fig  1b of the Hyperband thing
#
# # options, wall clock time or timesteps/checkpoints. validation, train or test loss
#
#
# Data
#
# Possibly generate some data to match pbt graphs from here in orion format: https://github.com/bkj/pbt
