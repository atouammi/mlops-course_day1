from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from data_loader import dataloader

def train(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df.iloc[:, :-1], df["species"], test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test

def get_accuracy(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    return accuracy

if __name__ == "__main__":
    iris_df = dataloader()
    model, X_train, X_test, y_train, y_test = train(iris_df)
    accuracy = get_accuracy(model, X_test, y_test)
    print(f"Accuracy is : {accuracy:.2f}")