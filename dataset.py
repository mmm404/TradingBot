import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """
    Load the CSV file into a DataFrame.
    """
    df = pd.read_csv(file_path)
    return df

def display_basic_info(df):
    """
    Display basic information and statistics of the DataFrame.
    """
    print("Basic Information:")
    print(df.info())
    
    print("\nStatistical Summary:")
    print(df.describe())
    
    print("\nMissing Values:")
    print(df.isnull().sum())

def plot_histograms(df):
    """
    Plot histograms for each column in the DataFrame.
    """
    df.hist(figsize=(15, 10))
    plt.suptitle('Histograms of All Columns')
    plt.show()

def plot_correlations(df):
    """
    Plot a heatmap of the correlation matrix.
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

def plot_pairplot(df):
    """
    Plot a pairplot for selected columns.
    """
    sns.pairplot(df)
    plt.title('Pairplot of Selected Columns')
    plt.show()

def plot_categorical_counts(df, categorical_columns):
    """
    Plot count plots for categorical variables.
    """
    for column in categorical_columns:
        plt.figure(figsize=(8, 6))
        sns.countplot(x=column, data=df)
        plt.title(f'Count Plot of {column}')
        plt.show()

def main():
    file_path = r'C:\Users\peter\working bot\calculated_indicators.csv'
    df = load_data(file_path)
    
    display_basic_info(df)
    # plot_histograms(df)
    # plot_correlations(df)
    # plot_pairplot(df)
    
    # Update with your categorical columns
    categorical_columns = ['column1', 'column2']
    # plot_categorical_counts(df, categorical_columns)
    
    print("EDA complete!")

if __name__ == "__main__":
    main()
