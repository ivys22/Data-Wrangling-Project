import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.sentiment_analysis.database import Session, SentimentAnalysisResult, RawComment, PreprocessedComment, SentimentSummary
from dagster import asset
import pandas as pd
from textblob import TextBlob

@asset
def raw_comments() -> pd.DataFrame:
    """Loads data from the database."""
    session = Session()
    query = session.query(RawComment).all()
    df = pd.DataFrame([{'comment_id': comment.comment_id, 'comment_text': comment.comment_text, 'is_poisonous': comment.is_poisonous} for comment in query])
    session.close()
    return df

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
    def analyze_sentiment(text):
        analysis = TextBlob(text)
        if analysis.sentiment.polarity > 0:
            return 'positive', analysis.sentiment.polarity
        elif analysis.sentiment.polarity == 0:
            return 'neutral', analysis.sentiment.polarity
        else:
            return 'negative', analysis.sentiment.polarity
    
    sentiments = preprocessed_comments['processed_text'].apply(lambda x: analyze_sentiment(x))
    return pd.DataFrame({
        "comment_id": preprocessed_comments["comment_id"],
        "sentiment": [s[0] for s in sentiments],
        "sentiment_score": [s[1] for s in sentiments]
    })

@asset
def sentiment_summary(sentiment_analysis: pd.DataFrame) -> pd.DataFrame:
    """Aggregates the sentiment analysis results to provide a summary. It groups the results by sentiment category ('positive', 'neutral', 'negative') and calculates the total count of comments and the average sentiment score within each category."""
    summary = sentiment_analysis.groupby('sentiment').agg(count=('sentiment', 'size'), average_score=('sentiment_score', 'mean')).reset_index()
    return summary