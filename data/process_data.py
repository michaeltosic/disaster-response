# Check the versions of libraries
import sys
print('Python: {}'.format(sys.version))

import numpy
print('numpy: {}'.format(numpy.__version__))

import pandas
print('pandas: {}'.format(pandas.__version__))

import sklearn
print('sklearn: {}'.format(sklearn.__version__))

from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    ## load categories dataset
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories)#, on="id")
    return(df)

def clean_data(df):
    
    print("Expanding categories column.")
    # create a dataframe of the 36 individual category columns and name the columns properly:
    categories = df.categories.str.split(";", expand = True) 
    row = categories.iloc[0]     # select the first row of the categories dataframe
    category_colnames = row.apply(lambda x: x[:len(x)-2])     # Slice string to remove last 2 characters from string
    categories.columns = category_colnames     # rename the columns of `categories`

    print("Converting category values to 0 or 1.")

    #convert category values to just numbers 0 or 1:
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1:]# set each value to be the last character of the string
        categories[column] = categories[column].astype(int)# convert column from string to numeric

    print("Wrapping up: Putting together the new clean df.")
    #wrap-up and concat
    df.drop("categories", inplace = True, axis = "columns")# drop the original categories column from `df`
    df = pd.concat([df, categories], axis = 1)# concatenate the original dataframe with the new `categories` dataframe
    
    print("Droping {} duplicates.".format(df.duplicated().sum()))

    #df.drop_duplicates()
    df.drop_duplicates(inplace=True)
    
    return df

def save_data(df, database_filename):
    pass  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()