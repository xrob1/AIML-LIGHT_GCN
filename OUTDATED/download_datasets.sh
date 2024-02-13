#!/bin/bash


mkdir data

mkdir ./data/facebook_book/ ./data/facebook_book/data
mkdir ./data/yahoo_movies ./data/yahoo_movies/data
mkdir ./data/movielens ./data/movielens/grouplens

curl -o ./data/facebook_book/data/dataset.tsv https://raw.githubusercontent.com/sisinflab/KGTORe/main/data/facebook_book/data/dataset.tsv

curl -o ./data/yahoo_movies/data/dataset.tsv https://raw.githubusercontent.com/sisinflab/KGTORe/main/data/yahoo_movies/data/dataset.tsv

curl -o ./data/movielens/grouplens/movies.dat https://raw.githubusercontent.com/sisinflab/KGTORe/main/data/movielens/grouplens/movies.dat
curl -o ./data/movielens/grouplens/ratings.dat https://raw.githubusercontent.com/sisinflab/KGTORe/main/data/movielens/grouplens/ratings.dat
curl -o ./data/movielens/grouplens/users.dat https://raw.githubusercontent.com/sisinflab/KGTORe/main/data/movielens/grouplens/users.dat
curl -o ./data/movielens/grouplens/README https://raw.githubusercontent.com/sisinflab/KGTORe/main/data/movielens/grouplens/README

