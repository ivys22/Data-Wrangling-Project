from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, Enum

Base = declarative_base()

class RawComment(Base):
    """Represents the table raw_comments with columns for comment_id (autoincrement primary key), comment_text (text of the comment), and is_poisonous (a boolean indicating whether the comment is harmful)."""
    __tablename__ = 'raw_comments'
    comment_id = Column(Integer, primary_key=True, autoincrement=True)
    comment_text = Column(String)
    is_poisonous = Column(Boolean)

class PreprocessedComment(Base):
    """Represents the table preprocessed_comments with columns for comment_id (primary key) and processed_text (the preprocessed text of the comment)."""
    __tablename__ = 'preprocessed_comments'
    comment_id = Column(Integer, primary_key=True)
    processed_text = Column(String)

class SentimentAnalysisResult(Base):
    """Represents the table sentiment_analysis with columns for comment_id (primary key), sentiment (an enum with values 'positive', 'neutral', 'negative'), and sentiment_score (a float representing the sentiment's intensity)."""
    __tablename__ = 'sentiment_analysis'
    comment_id = Column(Integer, primary_key=True)
    sentiment = Column(Enum('positive', 'neutral', 'negative', name='sentiment_types'))
    sentiment_score = Column(Float)

class SentimentSummary(Base):
    """Represents the table sentiment_summary with columns for sentiment (an enum, also serving as the primary key), count (an integer representing the number of comments with this sentiment), and average_score (a float representing the average sentiment score for this category)."""
    __tablename__ = 'sentiment_summary'
    sentiment = Column(Enum('positive', 'neutral', 'negative', name='sentiment_types'), primary_key=True)
    count = Column(Integer)
    average_score = Column(Float)

DATABASE_URI = 'sqlite:///mental_health_analysis.db'

engine = create_engine(DATABASE_URI)
Session = sessionmaker(bind=engine)

def init_db():
    """Initializes the database by creating all tables based on the defined schema. This setup ensures that the database is ready to accept data according to the defined structures."""
    Base.metadata.create_all(engine)