import os

FILE_PATH = os.path.abspath(__file__)
PROJECT_PATH = os.path.abspath(os.path.join(FILE_PATH, os.pardir, os.pardir))

# dataset
FACEBOOK = 'facebook_book'
YAHOO = 'yahoo_movies'
MOVIELENS = 'movielens'