from torch.utils.data import Dataset
from typing import List, Union
from .. import SentenceTransformer


class EncodeDataset(Dataset):
    def __init__(self,
                 sentences: Union[List[str], List[int]],
                 model: SentenceTransformer,
                 is_tokenized: bool = True, length_check: int = 128):
        """
        EncodeDataset is used by SentenceTransformer.encode method. It just stores
        the input texts and returns a tokenized version of it.
        """
        self.model = model
        self.sentences = sentences
        self.is_tokenized = is_tokenized
        self.length_check = length_check


    def __getitem__(self, item):
        return self.sentences[item][:self.length_check-2] if self.is_tokenized else self.model.tokenize(self.sentences[item])[:self.length_check-2]


    def __len__(self):
        return len(self.sentences)
