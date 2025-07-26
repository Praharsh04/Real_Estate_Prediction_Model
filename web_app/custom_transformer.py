from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

# Column indices based on the numerical attributes that will be passed to the pipeline
# The order is: ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # X is a numpy array here
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        
        if self.add_bedrooms_per_room:
            # Handle potential division by zero if total_rooms is zero
            with np.errstate(divide='ignore', invalid='ignore'):
                bedrooms_per_room = np.nan_to_num(X[:, bedrooms_ix] / X[:, rooms_ix])
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]