# ALTEGRAD Challenge 2021 : Predicting the h-index of authors
> Aymane BERRADI - Taoufik AGHRIS - Badr LAAJAJ

# Architecture of the project
## Folder 1: utils
It contains the different python files that include useful functions and classes.
## Folder 2: Data
It contains a text file of drive link to have all necessary data
## Notebooks Description :
* `Text_Preprocessing_and_Embbeding.ipynb`: it contains all preprocessing steps for text data, and embedding using Doc2Vec and S-BERT.
* `Node_embedding.ipynb`: it contains the embedding of the nodes using deepwalk for weighted and unweighted versions of the graph.
* `Features_engineering.ipynb`: it contains the structural features for graph data, and constructed variables used in the `Altegrad_Models.ipynb`.
* `Altegrad_Models.ipynb`: it contains several experiments on both features configuration and predictive models.
* `Kaggle_Submission.ipynb`: it's the notebook that reproduces predictions of our best performing model and lead us to the first place with score of 3.01562 on the private leaderboard https://www.kaggle.com/c/altegrad-2020/leaderboard.


