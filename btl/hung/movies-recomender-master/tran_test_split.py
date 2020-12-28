import pandas as pd
from sklearn.model_selection import train_test_split

col = ['userId', 'movieId', 'rating', 'timestamp']

data = pd.read_csv('ml-1m/ratings.dat', sep='::', names=col)
# data = pd.read_csv('ml-25m/ratings.csv')

train, test = train_test_split(data, test_size=0.2)
print(train)
train.sort_values('userId').to_csv('./ml-1m/train.csv', index=False)
test.sort_values('userId').to_csv('./ml-1m/test.csv', index=False)

