
import numpy as np
import pandas as pd

import sklearn
from sklearn.datasets import load_iris




def dataloader():
    # Load the iris dataset
    iris = load_iris()

    # Convert to a pandas DataFrame
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)


    # Add the target (species) to the DataFrame
    df['species'] = iris.target

    # If you want to see the first few rows
    return df






if __name__ == "__main__":

    iris_df = dataloader()
    print(iris_df.head())