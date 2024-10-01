from datasets import Dataset  
from functions import reverse_tokenization, tokenize_and_map_labels
from sklearn.base import BaseEstimator, TransformerMixin




class ReverseTokenizationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, reverse_map_tokens):
        self.reverse_map_tokens = reverse_map_tokens

    def fit(self, X, y=None):
        return self

    def transform(self, x):
        # Call reverse_tokenization with the correct parameters
        return reverse_tokenization(x, self.reverse_map_tokens)



class MapAndTokenizeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, tokenizer, map_tokens):
        self.tokenizer = tokenizer
        self.map_tokens = map_tokens

    def transform(self, examples):
        # Use the tokenizer from the instance variable
        return examples.map(lambda ex: tokenize_and_map_labels(ex, self.tokenizer, self.map_tokens), batched=True)

    def fit(self, X, y=None):
        return self  # No fitting required for this transformer

class DatasetFromListTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, x):
        return Dataset.from_list(x)

