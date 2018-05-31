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
import uuid

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


def _get_results_with_name(data, name):
    # TODO:

    all_data = []
    for data_point in data:
        concated = np.array([_get_feature_from_list(data_point['results'], name)])
        all_data.append(concated)

    Y = np.vstack(all_data)

    return Y

def main(arguments):

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('data', type=argparse.FileType('rb'), help="Input file as a pickle in Orion format.")
    parser.add_argument('--features', nargs="+", type=str, default=None, help="An optional list of features to use rather than the entire list.")
    parser.add_argument('--projection_type', choices=['2d', '3d', 'heatmap'], default='2d', help="The type of projection, a 2D plane, a 3D plane, or a heatmap.")
    parser.add_argument('--loss_interpolation_type', choices=["colour", "pointsize", "none"], help="How to show the validation loss value if at all.", default="colour")
    parser.add_argument("--phylo", action="store_true", default=False, help="If data contains phylogenetic-like information (what the ancestor of a hyperparameter configuration was, can draw connections. Otherweise, script interpolates connection from timestamp information.")
    parser.add_argument("--points", choices=["x", "o", "order", None], default="o", help="How to draw the individual data points. Order indicates should provide a number indicating in which order the configuration was run by start time.")
    parser.add_argument('--evaluation_metric', help="The evaluation metric to use for showing performance of features.")
    parser.add_argument('--title', default=None, help="How would you like to title your plot?")
    parser.add_argument('--save', action="store_true", default=False, help="Save the plot?")
    args = parser.parse_args(arguments)
    print(args)

    data = pickle.load(args.data)

    # currently only support real
    supported_feature_types = ['real']

    if not args.features:
        features = [x['name'] for x in data[0]["params"] if x['type'] in supported_feature_types]
        ignored_features = [x['name'] for x in data[0]["params"] if x['type'] not in supported_feature_types]
        print('Ignoring features {} because currently not supported. Only support {}.'.format(ignored_features, supported_feature_types))
    else:
        features = args.features

    if len(features) < 1:
        raise ValueError("Requires at least 2 features!")
    elif len(features) < 2 and args.projection_type == "3d":
        raise ValueError("To make a 3D graph need at least 2 features.")

    required_features = { "3d" : 2, "2d" : 1}

    sorted_data = sorted(data, key=lambda x: x['start_time'])

    X, projected = _get_matrix_from_features(data, features, required_features[args.projection_type])

    fig = plt.figure(figsize=(16, 8))

    # TODO: colours based on loss or size based on loss
    # TODO: handle non-numeric features
    if len(features) == required_features[args.projection_type]:
        # TODO: handle 3d projection
        plt.xlabel('Trials')
        plt.ylabel(features[0])
    else:
        plt.xlabel("Trials")
        plt.ylabel("PCA Component (Features {})".format(", ".join(features)))

    if args.loss_interpolation_type == "colour":
        Y = _get_results_with_name(data, args.evaluation_metric) #TODO: make configurable
        cm = plt.cm.get_cmap('RdYlBu')
        cmap=cm
        normalize = matplotlib.colors.Normalize(vmin=np.min(Y), vmax=np.max(Y))
        sc = plt.scatter(list(range(X.shape[0])), X[:,0], c=Y.reshape(-1), alpha=.8, norm=normalize)
        clbar = plt.colorbar(sc)
        clbar.set_label(args.evaluation_metric)
    elif args.loss_interpolation_type == "none":
        sc = plt.scatter(list(range(X.shape[0])), X[:,0], color="teal", alpha=.5)

    if args.title:
        plt.title(args.title)

    if args.save:
        unique_filename = str(uuid.uuid4())
        fig.savefig("{}.pdf".format(unique_filename), dpi=fig.dpi, bbox_inches='tight')
        print("Saved to: {}.pdf".format(unique_filename))
    else:
        plt.show()


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

# TODOs:
# 1. handle non-numeric data
# ~1hr
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
