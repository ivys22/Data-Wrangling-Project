from dagster import repository
from dagster import define_asset_job
from src.sentiment_analysis import assets, resources

@repository
def mental_health_repo():
    """Create jobs for each asset or asset group."""
    raw_comments_job = define_asset_job(
        name="raw_comments_job",
        selection=["raw_comments"]
    )

    preprocessed_comments_job = define_asset_job(
        name="preprocessed_comments_job",
        selection=["preprocessed_comments"]
    )

    sentiment_analysis_job = define_asset_job(
        name="sentiment_analysis_job",
        selection=["sentiment_analysis"]
    )

    sentiment_summary_job = define_asset_job(
        name="sentiment_summary_job",
        selection=["sentiment_summary"]
    )

    return [
        raw_comments_job,
        preprocessed_comments_job,
        sentiment_analysis_job,
        sentiment_summary_job
    ]
