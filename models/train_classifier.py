import sys
# import libraries
from sqlalchemy import create_engine
import pandas as pd
import re

#needed for text processing:
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def load_data(database_filepath):
    """ Function loads data from a SQL-database at the specified path
    INPUT
    SQL Database filepath
    OUTPUT
    X: features from database table
    y: labels from database table
    category_names: categories of labels
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table("Table", engine)
    X = df.message.values
    y = df.drop(["id","original","genre", "message", "index"], axis = "columns")
    category_names = set(y.columns)
    return (X, y, category_names)

def tokenize(text):
    pass


def build_model():
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()