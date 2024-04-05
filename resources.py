from dagster import resource, Field
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

class TextPreprocessor:
    """Defines a TextPreprocessor class that preprocesses text data to prepare it for NLP tasks."""
    def __init__(self, remove_stopwords=True):
        self.remove_stopwords = remove_stopwords
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = set(stopwords.words('english'))
    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        words = text.split()
        if self.remove_stopwords:
            words = [word for word in words if word not in self.stopwords]
        words = [self.lemmatizer.lemmatize(word) for word in words]
        return ' '.join(words)

@resource(config_schema={"remove_stopwords": Field(bool, is_required=False, default_value=True)})
def text_preprocessor_resource(init_context):
    """Defines a Dagster resource for the TextPreprocessor class, making it possible to integrate this text preprocessing utility within a Dagster pipeline."""
    ...