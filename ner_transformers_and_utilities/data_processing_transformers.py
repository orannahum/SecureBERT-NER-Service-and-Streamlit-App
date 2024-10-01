from datasets import Dataset  
from ner_transformers_and_utilities.functions import reverse_tokenization, tokenize_and_map_labels
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Dict, Any, Callable




class ReverseTokenizationTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer for reverse tokenization of predictions.

    Args:
        reverse_map_tokens (Dict[int, str]): A mapping from token IDs to their corresponding labels.
    """

    def __init__(self, reverse_map_tokens: Dict[int, str]) -> None:
        self.reverse_map_tokens = reverse_map_tokens

    def fit(self, X: List[List[int]], y: Any = None) -> "ReverseTokenizationTransformer":
        return self

    def transform(self, x: List[List[int]]) -> List[List[str]]:
        return reverse_tokenization(x, self.reverse_map_tokens)



class MapAndTokenizeTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that maps and tokenizes input examples using a tokenizer and a mapping of labels.

    Args:
        tokenizer (Callable): A tokenizer function that will be used to tokenize the input examples.
        map_tokens (Dict[str, int]): A mapping from token labels to their corresponding token IDs.
    """

    def __init__(self, tokenizer: Callable, map_tokens: Dict[str, int]) -> None:
        self.tokenizer = tokenizer
        self.map_tokens = map_tokens

    def transform(self, examples: Any) -> Any:
        return examples.map(lambda ex: tokenize_and_map_labels(ex, self.tokenizer, self.map_tokens), batched=True)

    def fit(self, X: Any, y: Any = None) -> "MapAndTokenizeTransformer":
        return self  # No fitting required for this transformer

class DatasetFromListTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that converts a list of examples into a Hugging Face Dataset.

    This transformer is useful for preparing data in a format compatible with Hugging Face datasets.

    Attributes:
        None
    """

    def __init__(self) -> None:
        pass

    def fit(self, X: Any, y: Any = None) -> "DatasetFromListTransformer":
        return self

    def transform(self, x: Any) -> Dataset:
        return Dataset.from_list(x)

