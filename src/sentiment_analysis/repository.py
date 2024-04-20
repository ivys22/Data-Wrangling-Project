from dagster import repository
from src.sentiment_analysis import assets
from src.sentiment_analysis import resources

@repository
def mental_health_repo():
    """Defines a repository which serves as a container for organizing and managing assets and resources for a data processing pipeline focused on mental health sentiment analysis."""
    return {
        "assets": [assets.raw_comments, assets.preprocessed_comments, assets.sentiment_analysis, assets.sentiment_summary],
        "resources": {"text_preprocessor": resources.text_preprocessor_resource},
    }