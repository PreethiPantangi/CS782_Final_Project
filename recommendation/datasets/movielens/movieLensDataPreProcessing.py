import zipfile
import pandas as pd
import os
from sklearn.model_selection import train_test_split

def process_movie_lens_data(path):
    extraction_path = "./recommendation/datasets/movielens/"

    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(extraction_path)

    ratings_path = "./recommendation/datasets/movielens/ml-1m/ratings.dat"

    ratings = pd.read_csv(ratings_path, sep='::', header=None, engine='python')
    ratings.to_csv('./recommendation/datasets/movielens/movieLens.txt', index=False, header=False, sep='\t')
    split_test_and_train(ratings)

def split_test_and_train(ratings):
    train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

    # Create folders if they don't exist
    folders = ['test', 'validation']
    for folder in folders:
        os.makedirs(f'./recommendation/datasets/movielens/{folder}', exist_ok=True)

    # Write train and test data to corresponding folders
    train_data.to_csv('./recommendation/datasets/movielens/test/train.txt', index=False, header=False, sep='\t')
    test_data.to_csv('./recommendation/datasets/movielens/test/test.txt', index=False, header=False, sep='\t')

    # Copy train.txt and test.txt to validation folder
    os.system('cp ./recommendation/datasets/movielens/test/train.txt ./recommendation/datasets/movielens/validation/train.txt')
    os.system('cp ./recommendation/datasets/movielens/test/test.txt ./recommendation/datasets/movielens/validation/test.txt')

# This is a guard clause that ensures the code is executed only if the script is run directly, 
# and not when it's imported by another script.
if __name__ == "__main__":
    process_movie_lens_data("ml-1m.zip")
