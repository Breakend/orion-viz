# Welcome to the feature visualization utilities

Let us walk through the possibilities with examples...

## Basic feature plot

First, we can use a generated dataset from Orion of hyperparameter configurations to generate a simple plot without specifying features:

```
python feature_plots.py ../data/rosenbrock-5.pkl --loss_interpolation none --evaluation_metric rosenbrock --projection_type 2d --save
```

![rosenbrock5basic](images/rosenbrock5-basic.pdf?raw=true "Title")

As can be seen, since there are many hyperparameters in the search space, the script has automatically inferred a PCA space to plot the distribution of trials.


## 3D Projection

We can also do this in 3D:

```
python3 feature_plots.py ../data/rosenbrock-5.pkl --loss_interpolation none --evaluation_metric rosenbrock --projection_type 3d --save
```

![rosenbrock5basic3d](images/rosenbrock5basic3d.pdf?raw=true "Title")

## Adding a title

and we can add a title

```
python3 feature_plots.py ../data/rosenbrock-5.pkl --loss_interpolation none --evaluation_metric rosenbrock --title MyHyperparams --projection_type 3d --save
```

![rosenbrock5basic3d](images/rosenbrock3dwithtitle.pdf?raw=true "Title")


## Colouring by evaluation metric

We can colour the scatter points by an evaluation metric:

```
python3 feature_plots.py ../data/rosenbrock-5.pkl --loss_interpolation colour --evaluation_metric rosenbrock --title MyHyperparams --projection_type 3d --save
```

![rosenbrock5basic3d](images/lossinterpolationrosenbrockcolour.pdf?raw=true "Title")

## Colouring by time

Or we can colour by time as well:

```
python3 feature_plots.py ../data/A.pkl --loss_interpolation colour --evaluation_metric valid_m --use_time --time_axis colour --title MyHyperparams --projection_type 3d --save
```


![timeascolor](images/timeascolour.pdf?raw=true "Title")

## Or we can make time the X axis

```
python3 feature_plots.py ../data/A.pkl --loss_interpolation colour --evaluation_metric valid_m --use_time --time_axis colour --title MyHyperparams --projection_type 3d --save
```

![timeasx](images/timexaxis.pdf?raw=true "Title")

## Or we can make time both the x axis and colour

```
python3 feature_plots.py ../data/A.1.pkl --loss_interpolation colour --evaluation_metric valid_m --use_time --time_axis X+colour --title MyHyperparams --projection_type 3d --save
```

![timeasxcolor](images/timexandcolour.pdf?raw=true "Title")


## Maybe you want to just add labels to the points instead to indicate time/order

```
python3 feature_plots.py ../data/rosenbrock-5.pkl --loss_interpolation colour --evaluation_metric rosenbrock --point_label order --projection_type 2d --save --fig_size 32 16
```

![timeasxcolor](images/orderlabels.pdf?raw=true "Title")

## Maybe you only care about comparing certain features

You can specify the features you want. Anything greater than the number of axes you have available will run PCA and compress the dimensions.

```
python3 feature_plots.py ../data/rosenbrock-5.pkl --loss_interpolation colour --evaluation_metric rosenbrock --projection_type 2d --save --features a b
```

![timeasxcolor](images/specificfeatures.pdf?raw=true "Title")

## Or you want a heatmap made out of those features

```
python3 feature_plots.py ../data/rosenbrock-5.pkl --loss_interpolation heatmap --evaluation_metric rosenbrock --projection_type 2d --save --features a b
```

![timeasxcolor](images/heatmap.pdf?raw=true "Title")
