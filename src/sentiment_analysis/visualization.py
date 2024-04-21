import pandas as pd
import matplotlib.pyplot as plt
from dagster import asset
from textblob import TextBlob

def analyze_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'positive', analysis.sentiment.polarity
    elif analysis.sentiment.polarity == 0:
        return 'neutral', analysis.sentiment.polarity
    else:
        return 'negative', analysis.sentiment.polarity

@asset
def sentiment_analysis(preprocessed_comments: pd.DataFrame) -> pd.DataFrame:
    sentiments = preprocessed_comments['processed_text'].apply(analyze_sentiment)
    result_df = pd.DataFrame(sentiments.tolist(), columns=["sentiment", "sentiment_score"], index=preprocessed_comments.index)
    result_df["comment_id"] = preprocessed_comments["comment_id"]
    return result_df

@asset
def sentiment_summary(sentiment_analysis: pd.DataFrame) -> pd.DataFrame:
    summary = sentiment_analysis.groupby('sentiment').agg(count=('sentiment', 'size'), average_score=('sentiment_score', 'mean')).reset_index()
    return summary

def plot_sentiment_summary(summary: pd.DataFrame):
    fig, ax = plt.subplots()
    summary.plot(kind='bar', x='sentiment', y='count', ax=ax, color=['green', 'blue', 'red'])
    ax.set_title('Sentiment Classification of Mental Health Comments')
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Count of Comments')
    plt.xticks(rotation=0)
    plt.show()