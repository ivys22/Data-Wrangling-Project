import pandas as pd
from src.sentiment_analysis import assets
from src.sentiment_analysis.assets import raw_comments
from dagster import build_op_context
from unittest.mock import patch, Mock, MagicMock

def test_raw_comments():
    """Tests the raw_comments function."""
    mock_data = pd.DataFrame({
        'text': ["Good service", "Bad experience"],
        'label': [0, 1]
    })

    with patch('src.sentiment_analysis.assets.pd.read_csv', return_value=mock_data):
        result = raw_comments()

        assert isinstance(result, pd.DataFrame), "The result should be a DataFrame"
        assert not result.empty, "The result DataFrame should not be empty"
        assert len(result) == 2, "DataFrame should contain two records"
        assert list(result.columns) == ["text", "label"], "DataFrame should have the correct columns"

def test_preprocessed_comments():
    raw_comments_df = pd.DataFrame({
        "comment_id": [1, 2],
        "comment_text": ["Test comment", "Another comment"],
        "is_poisonous": [0, 1]
    })

    mock_resource = Mock()
    mock_resource.preprocess.side_effect = lambda x: x.upper()

    context = build_op_context(resources={"text_preprocessor": mock_resource})

    result = assets.preprocessed_comments(context, raw_comments_df)

    
def test_sentiment_analysis():
    """Tests the sentiment_analysis function."""
    preprocessed_comments_df = pd.DataFrame({
        "comment_id": [1, 2],
        "processed_text": ["Sad right now", "Happy right now"]
    })

    result = assets.sentiment_analysis(preprocessed_comments_df)
    assert not result.empty, "The result DataFrame should not be empty"
    assert all(result['sentiment'].isin(['positive', 'neutral', 'negative'])), "Each sentiment should be categorized correctly"

def test_sentiment_summary():
    """Tests the sentiment_summary function."""
    sentiment_df = pd.DataFrame({
        "comment_id": [1, 2, 3],
        "sentiment": ["positive", "negative", "positive"],
        "sentiment_score": [0.5, -0.25, 0.75]
    })

    result = assets.sentiment_summary(sentiment_df)
    assert not result.empty, "The result should not be empty"
    assert len(result) == 2, "There should be two sentiment categories in the summary"
    assert all(column in result.columns for column in ['sentiment', 'count', 'average_score']), "Summary should have all required columns"
