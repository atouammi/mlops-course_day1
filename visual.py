import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def plot_feature(df, feature):
    # Plot a histogram of one of the features
    df[feature].hist()
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.show()

def plot_features(df):
    # Plot scatter plot of first two features.
    scatter = plt.scatter(
        df["sepal length (cm)"], df["sepal width (cm)"], c=df["species"]
    )
    plt.title("Scatter plot of the sepal features (width vs length)")
    plt.xlabel(xlabel="sepal length (cm)")
    plt.ylabel(ylabel="sepal width (cm)")
    plt.legend(
        scatter.legend_elements()[0],
        df["species_name"].unique(),
        loc="lower right",
        title="Classes",
    )
    plt.show()

def plot_model(model, X_test, y_test):
    # Plot the confusion matrix for the model
    ConfusionMatrixDisplay.from_estimator(estimator=model, X=X_test, y=y_test)
    plt.title("Confusion Matrix")
    plt.show()




if __name__ == "__main__":
    iris_df = load_dataset()
    model, X_train, X_test, y_train, y_test = train(iris_df)
    accuracy = get_accuracy(model, X_test, y_test)
    print(f"Accuracy: {accuracy:.2f}")

    plot_feature(iris_df, "sepal length (cm)")
    plot_features(iris_df)
    plot_model(model, X_test, y_test)