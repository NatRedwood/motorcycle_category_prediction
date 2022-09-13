# Motorcycle Category Prediction

The repo includes different ML models trained of motorcycle specifications data to predict the motorcycle category.

## Dataset

The preprocessed data was downloaded from the GitHub repository: https://github.com/rsc22/Motocrcycle_Classification/tree/main/data

## Description

The pdf file [project_pdf](https://github.com/Nwojarnik/motorcycle_category_prediction/blob/main/motorcycle_category_prediction/scripts_with_processed_data/Motorcycle_classification.pdf) includes the analysis and interpretation of the model results.

## Project conclusions in a nutshell:
* the NN model has 43 features which are different motorcycle specifications, such as rear wheel width, torque, number of disks at the front or rear brake, number or cylinders or engine size (ccm).
* first 9 models explore the peformance based on different activation functions and different number of hidden layers
* further exploration refers to the increase of the number of hidden layers and deceasing the learning rate
* the best model (with 75 hidden layers and the learning rate 0.000001) is further explored by splitting the training data into 20,40, 60 and 80%
* the main conclusions relate to the fact that the large dataset requires the bigger size of model in general; the number of features (43) makes the model more robust and forces the model to be trained on more complex parameters (such as 20000 iterations); the model performs significantly better when the number of hidden layers is increased which means that the complexity of the dataset requires from the model high density.

## Project visualizations

The folder [plots](https://github.com/Nwojarnik/motorcycle_category_prediction/tree/main/motorcycle_category_prediction/plots) includes the heatmaps. Most of them are obvious and reasonably explainable like dry weight, ccm and rear tyre width.
![motorcycle specifications correlations](https://github.com/Nwojarnik/motorcycle_category_prediction/blob/main/motorcycle_category_prediction/plots/corr_all_feats.png)


Other tables such as explore the performance of different activation functions and model parameters:
![plot of 9 trained models](https://github.com/Nwojarnik/motorcycle_category_prediction/blob/main/motorcycle_category_prediction/plots/results_9models.png)

## Dedications

I dedicate this project to my one and only love: 
![Moto Guzzi III V7 Stone](https://github.com/Nwojarnik/motorcycle_category_prediction/blob/main/motorcycle_category_prediction/one_and_only_love.jpg)
