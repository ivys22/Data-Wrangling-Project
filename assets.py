from dagster import asset
import pandas as pd
from textblob import TextBlob

@asset
def raw_comments() -> pd.DataFrame:
    """Defines an asset raw_comments that reads a CSV file into a Pandas DataFrame."""
    return pd.read_csv("data/mental_health.csv")

@asset(required_resource_keys={"text_preprocessor"})
def preprocessed_comments(context, raw_comments: pd.DataFrame) -> pd.DataFrame:
    """Takes the raw comments and preprocesses them using a text preprocessing resource."""
    processed_texts = raw_comments['comment_text'].apply(lambda text: context.resources.text_preprocessor.preprocess(text))
    return pd.DataFrame({
        "comment_id": raw_comments["comment_id"],
        "processed_text": processed_texts,
        "is_poisonous": raw_comments["is_poisonous"]
    })

@asset
def sentiment_analysis(preprocessed_comments: pd.DataFrame) -> pd.DataFrame:
    """Performs sentiment analysis on the preprocessed comments. For each comment, it uses TextBlob to calculate the sentiment polarity. Based on the polarity, comments are categorized as 'positive', 'neutral', or 'negative'."""
    ...

@asset
def sentiment_summary(sentiment_analysis: pd.DataFrame) -> pd.DataFrame:
    """Aggregates the sentiment analysis results to provide a summary. It groups the results by sentiment category ('positive', 'neutral', 'negative') and calculates the total count of comments and the average sentiment score within each category."""
    ...