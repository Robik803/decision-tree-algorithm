import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump
from sklearn import tree
from sklearn.feature_selection import SelectKBest

TARGET = "https://raw.githubusercontent.com/4GeeksAcademy/decision-tree-project-tutorial/main/diabetes.csv"

param_grid = {
    "max_depth": [3, 5, 10, 15, None],
    "min_samples_split": [2, 5, 10, 15],
    "min_samples_leaf": [1, 2, 4, 8],
    "criterion": ["gini", "entropy"]
}

def read_data() -> pd.DataFrame:
    """Load dataset from URL"""
    try:
        return pd.read_csv(TARGET)
    except Exception as e:
        print("Error reading the file:", e)
        return None
    
def save_data(df: pd.DataFrame, path: str) -> None:
    """Save DataFrame to CSV"""
    try:
        df.to_csv(path, index=False)
    except Exception as e:
        print("Error writing the file:", e)

def display_summary(y_test, y_pred):
    print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


def main():

    # Load dataset
    df = read_data()
    if df is None:
        return
    
    df.head()
    df.tail()
    df.info()
    df.describe()

    # Remove duplicates
    if df.duplicated().sum() > 0:
        print("There are duplicated rows\n")
        df.drop_duplicates(inplace=True)
    else:
        print("There are no duplicated rows\n") 

    # Check and handle missing values
    if df.isnull().sum().sum() > 0:
        print("There are missing values\n")
        df.dropna(inplace=True)
    else:
        print("There are no missing values\n")
    
    # Feature transformations
    df["DiabetesPedigreeFunction"] = np.log1p(df["DiabetesPedigreeFunction"])
    print("After cleaning the data\n", df.describe())

    # Save cleaned data
    save_data(df, "./data/interim/clean_diabetes_data.csv")

    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()
    
    # Split dataset
    X = df.drop(columns="Outcome")
    y = df["Outcome"]

    # Parallel coordinates plot
    total_data = df.copy()
    pd.plotting.parallel_coordinates(total_data, "Outcome", color=("#E58139", "#39E581", "#8139E5"))
    plt.title("Parallel Coordinates Plot")
    plt.show()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    select_k_best = SelectKBest(k=8)
    select_k_best.fit_transform(X_train, y_train)
    selected_columns = X.columns[select_k_best.get_support()]
    X_train_selected = pd.DataFrame(select_k_best.transform(X_train), columns=selected_columns)
    X_test_selected = pd.DataFrame(select_k_best.transform(X_test), columns=selected_columns)
    save_data(X_train_selected, "./data/processed/X_train_selected.csv")
    save_data(X_test_selected, "./data/processed/X_test_selected.csv")
    save_data(y_train, "./data/processed/y_train.csv")
    save_data(y_test, "./data/processed/y_test.csv")
    print("X_train_selected:\n", X_train_selected.head())

    # Train Decision Tree Model
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train_selected, y_train)

    # Plot decision tree
    plt.figure(figsize=(15, 10))
    tree.plot_tree(model, feature_names=X_train_selected.columns, filled=True)
    plt.title("Decision Tree Structure")
    plt.show()

    # Evaluate model
    y_pred = model.predict(X_test)
    display_summary(y_test, y_pred)

    # Hyperparameter tuning with GridSearchCV
    grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=10, scoring="accuracy")
    grid_search.fit(X_train_selected, y_train)
    print("\nBest Parameters:", grid_search.best_params_)
    best_model = DecisionTreeClassifier(random_state=42, **grid_search.best_params_)
    best_model.fit(X_train_selected, y_train)
    
    # Plot optimized decision tree
    plt.figure(figsize=(15, 10))
    tree.plot_tree(best_model, feature_names=X_train_selected.columns, filled=True)
    plt.title("Optimized Decision Tree Structure")
    plt.show()
    y_pred = best_model.predict(X_test)
    print("\nOptimized Model")
    display_summary(y_test, y_pred)
    dump(best_model, "./data/processed/model_dt_entropy_d5_l8_s2.sav")

if __name__ == '__main__':
    main()
