
# coding: utf-8
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import cross_validation #split test and train set
import graphlab

anime = pd.read_csv('C:/Users/kexic/Documents/AI/anime-recommendations-database/anime_new.csv') # load the data
rate = pd.read_csv('C:/Users/kexic/Documents/AI/anime-recommendations-database/rating.csv') # load the data

print anime.head()

print rate.head()

anime = anime.drop('genre', 1)
rate= rate.drop(rate[rate.rating == -1].index)

print anime.head()

print rate.head()

r_train, r_test = cross_validation.train_test_split(rate, train_size=0.8)
print r_train.shape
print r_test.shape

train_data = graphlab.SFrame(r_train)
test_data = graphlab.SFrame(r_test)


popularity_model = graphlab.popularity_recommender.create(train_data, user_id='user_id', item_id='anime_id', target='rating')


#Get recommendations for first 5 users and print them
#users = range(1,6) specifies user ID of first 5 users
#k=5 specifies top 5 recommendations to be given
popularity_recomm = popularity_model.recommend(users=range(1,6),k=5)
popularity_recomm.print_rows(num_rows=25)


#Train Model(pearson)
item_sim_model = graphlab.item_similarity_recommender.create(train_data, user_id='user_id', item_id='anime_id', target='rating', similarity_type='pearson')
item_sim_recomm = item_sim_model.recommend(users=range(1,6),k=5)
item_sim_recomm.print_rows(num_rows=25)


#Campare popular model vs. pearson model
model_performance = graphlab.compare(test_data, [popularity_model, item_sim_model])
graphlab.show_comparison(model_performance,[popularity_model, item_sim_model])


#Train Model(Jaccard)
item_sim_model2 = graphlab.item_similarity_recommender.create(train_data, user_id='user_id', item_id='anime_id', target='rating', similarity_type='jaccard')
item_sim_recomm2 = item_sim_model2.recommend(users=range(1,6),k=5)
item_sim_recomm2.print_rows(num_rows=25)


#campare 3 models
model_performance3 = graphlab.compare(test_data, [popularity_model,item_sim_model,item_sim_model2])
graphlab.show_comparison(model_performance3,[popularity_model,item_sim_model,item_sim_model2])


#Train Model(Cosine)
item_sim_model3 = graphlab.item_similarity_recommender.create(train_data, user_id='user_id', item_id='anime_id', target='rating', similarity_type='cosine')
item_sim_recomm3 = item_sim_model3.recommend(users=range(1,6),k=5)
item_sim_recomm3.print_rows(num_rows=25)


#campare 4 models
model_performance4 = graphlab.compare(test_data, [popularity_model,item_sim_model,item_sim_model2,item_sim_model3])
graphlab.show_comparison(model_performance4,[popularity_model,item_sim_model,item_sim_model2,item_sim_model3])

