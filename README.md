# Sentiment Analysis on Mental Health Comments

## Project Overview

This project is focused on analyzing sentiment in comments related to mental health issues, utilizing a dataset from Kaggle titled "Mental Health Corpus." The primary goal is to understand the language and sentiment surrounding discussions on mental health. The project involves extracting raw comments from the dataset, preprocessing the text data, performing sentiment analysis using the TextBlob library, and storing the results in a database for further exploration and reporting. 

## Setup Instructions

### Prerequisites

* Python 3.10.13
* Pip (Python package installer)
* SQLite (for the default database setup)

### Initial Configuration

* NLTK Data: Download the required NLTK datasets: 

    *import nltk*
    *nltk.download('stopwords')*
    *nltk.download('wordnet')*
    *nltk.download('omw-1.4')*

* Database Setup: Initialize the database using SQLAlchemy:

    *from src.sentiment_analysis.database import init_db*
    *init_db()*

## Project Structure

/src
    /sentiment_analysis
        assets.py # Dagster assets for the ETL pipeline
        database.py # SQLAlchemy ORM models and database initialization
        repository.py # Dagster repository to orchestrate jobs
        resources.py # Resources for text preprocessing
        visualization.py # Dash application for data visualization
    /tests
        test_sentiment.py # Pytest unit tests for functionality
    /data
        mental_health.csv # Dataset file

## Running the Project

To execute the Dagster pipeline and start the Dash visualization for sentiment analysis results:

* Run the Dagster pipeline:

    dagit -f src/sentiment_analysis/repository.py

Access the Dagit UI via a web browser to visualize and manage the pipeline.

* To start the Dash web application:

    python src/sentiment_analysis/visualization.py

## Testing

The project uses pytest for testing. To run the tests, execute the following in the project directory:

    pytest
    
