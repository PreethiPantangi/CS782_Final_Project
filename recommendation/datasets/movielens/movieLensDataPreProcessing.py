import zipfile
import pandas as pd

zip_file_path = "ml-1m.zip"
extraction_path = "ml-1m"

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extraction_path)

ratings_path = "ml-1m/ml-1m/ratings.dat"

rating_cols = ['UserID', 'Movie ID', 'Rating', 'Timestamp']
ratings = pd.read_csv(ratings_path, sep='::', header=None, engine='python')
ratings.to_csv('movieLens.txt', index=False, header=False, sep='\t')
