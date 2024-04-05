from dagster import repository
from assets import raw_comments, preprocessed_comments, sentiment_analysis, sentiment_summary
from resources import text_preprocessor_resource

@repository
def mental_health_repo():
    """Defines a repository which serves as a container for organizing and managing assets and resources for a data processing pipeline focused on mental health sentiment analysis."""
    return {
        "assets": [raw_comments, preprocessed_comments, sentiment_analysis, sentiment_summary],
        "resources": {"text_preprocessor": text_preprocessor_resource},
    }