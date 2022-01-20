# import libraries
import sys
import pandas as pd

#sqldatabase:
from sqlalchemy import create_engine

#regex:
import re

#needed for text processing:
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

#for pipeline:
from sklearn.pipeline import Pipeline

#for estimators:
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

#for training:
from sklearn.model_selection import train_test_split

#for testing:
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#for Grid search:
from sklearn.model_selection import GridSearchCV


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
    """Function for text processing, in particular it replaces urls, tokenizes and lemmatizes the words in a given text.
    INPUT
    text: text to process as str
    OUTPUT:
    tokens: list of tokenized words"""
    
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
 
    # normalize case and remove punctuation using regex
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text with the tokenizer
    tokens = word_tokenize(text)
    
    # lemmatize and remove stop words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words("english")]

    return tokens

#model definition
def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    return pipeline

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