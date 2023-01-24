from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import numpy as np

class RandomDelete:
    def __init__(self, deletion_rate=.1):
        """Augmentation for NLP models, delete random words with given probability

        Parameters
        ----------
        deletion_rate : float, optional
            probability for word deletion, by default .1
        """
        self.deletion_rate = deletion_rate

    def __call__(self, sample: str) -> str:
        """Removes random words from sample

        Parameters
        ----------
        sample : str
            input text

        Returns
        -------
        str
            text missing words
        """
        
        words = list(set(word_tokenize(sample)))
        chosen_words = np.where(np.random.binomial(1, self.deletion_rate, size=len(words)) == 1)[0]
        
        for chosen_word in chosen_words:
            sample = sample.replace(words[chosen_word], "")
        
        return sample