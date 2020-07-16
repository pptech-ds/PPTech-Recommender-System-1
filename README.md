# About this project
To understand recommender system I followed the tutorial from "https://stackabuse.com/creating-a-simple-recommender-system-in-python-using-pandas"
Database used for this tuto comming from "https://grouplens.org/datasets/movielens/"
For this project I will use the small one to ease data processing "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

# Step by Step
### 1- Reading data containing movie ratings
```python
ratings_data = pd.read_csv("ratings_small.csv")
print('--- Reading data containing movie ratings ---')
print(ratings_data.head())
print('----------------------\n\n')
```
```python
--- Reading data containing movie ratings ---
    userId  movieId  rating  timestamp
0        1        1     4.0  964982703
1        1        3     4.0  964981247
2        1        6     4.0  964982224
3        1       47     5.0  964983815
4        1       50     5.0  964982931
----------------------
```
