
from sklearn.datasets import load_iris

from sklearn.linear_model import LogisticRegression


from sklearn.model_selection import train_test_split


from sklearn.metrics import accuracy_score




model = LogisticRegression()
def train_eval(df,model):
    
    X = df.drop('species',axis=1)
    y = df.species

    X_train , X_test , y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    

    model.fit(X_train,y_train)


    y_pred = model.predict(X_test)


    acc  = accuracy_score(y_pred,y_test)
    print(f"accuracy_score is :{acc}")
