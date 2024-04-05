import pandas as pd
from src import assets
from unittest.mock import patch
from unittest.mock import Mock

def test_raw_comments():
    """Tests the raw_comments function."""
    mock_data = pd.DataFrame({
        "comment_id": [1, 2],
        "comment_text": ["Good service", "Bad experience"],
        "is_poisonous": [0, 1]
    })

    with patch('pandas.read_csv', return_value=mock_data):
        result = assets.raw_comments()
        assert isinstance(result, pd.DataFrame), "The result should be a DataFrame"
        assert not result.empty, "The result DataFrame should not be empty"
        assert list(result.columns) == ["comment_id", "comment_text", "is_poisonous"], "DataFrame should have the correct columns"

def test_preprocessed_comments():
    """Tests the preprocessed_comments function."""
    raw_comments_df = pd.DataFrame({
        "comment_id": [1, 2],
        "comment_text": ["Test comment", "Another comment"],
        "is_poisonous": [0, 1]
    })

    mock_context = Mock()
    mock_context.resources.text_preprocessor.preprocess.side_effect = lambda x: x.upper() 

    result = assets.preprocessed_comments(mock_context, raw_comments_df)
    assert not result.empty, "The result should not be empty"
    assert all([isinstance(text, str) for text in result['processed_text']]), "All processed texts should be strings"
    assert result['processed_text'][0] == "TEST COMMENT", "Text should be processed correctly"
    
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
    ...
