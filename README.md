# Disaster Response Pipeline Project

### Description:
The project disaster response show the use of machine learning for the classification of messages in the case of a disaster. In such instances the authorities receive messages over different channels and it is also at a time where their capacity is limited. The use of machine leaning to classifiy the messages help to match the messages to the correct departements that can provide aid (being it police, medical services, food, shelter, etc.).

This repo is to be submitted as part of a project assignment in a data science course.


### Project Structure:
The project is made out of the following files:
* data/process_data.py: script gathers messages (disaster_messages.csv) and categories (disaster-categories.csv) provided in two csv files in the folder data. It loads the data into a single dataframe and cleans it so that that the end result is a dataframe that could be used for machine learning. The resulting dataframe is saved in a SQL database (DisasterResponse.db).
* models/train_classifier.py: script loads the data from the SQL database and prepares a machine learning pipeline for text processing and multi output classification. The text documents are  tokenized and lemmanized and put into a matrix of tokens. The matrix is used to create a normalized tf-idf representation which is then fed into a multi output classifier. Gridsearch is used to demonstrate parameter tuning. The classifier is finally saved in a pkl-file.
* app/run.py: flask app that demonstrates the working classifier. New messages can be added and they are categorized accordingly.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage