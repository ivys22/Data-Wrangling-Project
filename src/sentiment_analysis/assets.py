import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.sentiment_analysis.database import Session, SentimentAnalysisResult, RawComment, PreprocessedComment, SentimentSummary, engine
from kaggle.api.kaggle_api_extended import KaggleApi
from dagster import asset, build_op_context
import pandas as pd
from textblob import TextBlob

@asset
def raw_comments() -> pd.DataFrame:
    """Loads data from the mental_health.csv file into the database."""
    session = Session()
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files('reihanenamdari/mental-health-corpus', path='data', unzip=True)
    df = pd.read_csv('data/mental_health.csv')
    print("Columns in loaded DataFrame:", df.columns)
    try:
        for _, row in df.iterrows():
            comment = RawComment(
                comment_text=row['text'], 
                is_poisonous=bool(row['label']) 
            )
            session.add(comment)
        session.commit()
    except Exception as e:
        session.rollback()
    finally:
        session.close()
    return df

@asset(required_resource_keys={"text_preprocessor"})
def preprocessed_comments(context, raw_comments: pd.DataFrame) -> pd.DataFrame:
    """Takes the raw comments and preprocesses them using a text preprocessing resource."""
    if 'text' not in raw_comments.columns:
        raise ValueError("DataFrame does not contain 'text' column")
    if 'label' not in raw_comments.columns:
        raise ValueError("DataFrame does not contain 'label' column")
    processed_texts = raw_comments['text'].apply(lambda text: context.resources.text_preprocessor.preprocess(text))
    return pd.DataFrame({
        "comment_id": raw_comments.index,
        "processed_text": processed_texts,
        "is_poisonous": raw_comments["label"]
    })

@asset
def sentiment_analysis(preprocessed_comments: pd.DataFrame) -> pd.DataFrame:
    """Performs sentiment analysis on the preprocessed comments using TextBlob."""
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
    """Aggregates results to provide a sentiment summary."""
    summary = sentiment_analysis.groupby('sentiment').agg(count=('sentiment', 'size'), average_score=('sentiment_score', 'mean')).reset_index()
    return summary
if __name__ == '__main__':
    class MockTextPreprocessor:
        def preprocess(self, text):
            return text.lower()
    context = build_op_context(resources={'text_preprocessor': MockTextPreprocessor()})
    raw_comments_df = raw_comments()
    preprocessed_comments_df = preprocessed_comments(context, raw_comments=raw_comments_df)
    sentiment_df = sentiment_analysis(preprocessed_comments=preprocessed_comments_df)
    summary_df = sentiment_summary(sentiment_df)
    print(summary_df)

@asset
def emotion_analysis(preprocessed_comments: pd.DataFrame) -> pd.DataFrame:
    """Performs emotion analysis on the preprocessed comments using NRCLex."""