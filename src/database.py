from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class RawComment(Base):
    """Represents the table raw_comments with columns for comment_id (primary key), comment_text (text of the comment), and is_poisonous (a boolean indicating whether the comment is harmful)."""
    ...

class PreprocessedComment(Base):
    """Represents the table preprocessed_comments with columns for comment_id (primary key) and processed_text (the preprocessed text of the comment)."""
    ...

class SentimentAnalysisResult(Base):
    """Represents the table sentiment_analysis with columns for comment_id (primary key), sentiment (an enum with values 'positive', 'neutral', 'negative'), and sentiment_score (a float representing the sentiment's intensity)."""
    ...

class SentimentSummary(Base):
    """Represents the table sentiment_summary with columns for sentiment (an enum, also serving as the primary key), count (an integer representing the number of comments with this sentiment), and average_score (a float representing the average sentiment score for this category)."""
    ...

DATABASE_URI = 'sqlite:///mental_health_analysis.db'

engine = create_engine(DATABASE_URI)
Session = sessionmaker(bind=engine)

def init_db():
    """When called, will create all tables in the database based on the schema defined by the model classes inheriting from 'Base'."""
    ...