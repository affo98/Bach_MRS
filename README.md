# Bach_MRS
"Recommender Systems have accurately learned to recommend items based on im-
plicit feedback such as; mouse hovering, number of items in online shopping basket,
or what percentage of a movie is finished on a streaming service [18]. This feedback
can be converted directly into revenue by accurately predicting the next preferred
item of a user, however, researchers have stated the vitality of evaluating recom-
mender systems on novelty and diversity for optimised performance in a modern
recommender systems[3]. This project investigates the ability to include novelty
and diversity in item recommendation in a music recommender setting. The project
uses the Million Playlist Dataset that was published following the ACM RecSys
Challenge 2018[4]. By deploying an evaluation framework using the python libraries
Lenskit and Recbole this paper compares 6 different models on their ability to in-
clude novelty defined by the Catalogue Coverage, and the diversity defined by the
Gini Index. The models compared in this project are : Most Popular, Matrix
Factorization from Implict Feedback, Biased SVD, Bayesian Personalised Ranking,
Neuro Collaborative Filtering, and Neuro Attentive Item Similarity. The model
performance is also evaluated on the accuracy metrics : NDCG@10, Recall@10,
Precision@10 and Hit@10, to study the diversity-accuracy tradeoff [13]. The paper
found that model giving the most diverse and novel recommendations, was the Neu-
ral Collaborative Filtering implemented with RecBole[12, 24], that best learned the
user item relationship with its non-linearity and item-features."

The code is structured with three main folders: "general", "recBole", "lenskit". The General folder is comprised of all data processing,
filtering and visualisations, whereas the folder recBole and Lenskit are the implemented models utilised in this project.

Findings of model performances:
![heatmap](https://github.com/affo98/Bach_MRS/assets/90624056/04160943-5188-41b0-85a1-786dfe0cfe1d)




