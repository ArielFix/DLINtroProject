from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import random

class SynonymReplace:
    def __init__(self, number_of_replacements=5):
        """Augmentation for NLP based models, replaces words with synonyms

        Parameters
        ----------
        number_of_replacements : int, optional
            number of words to replace per sentence, by default 5
        """
        self.number_of_replacements = number_of_replacements
    def __call__(self, sample: str) -> str:
        """Replace words in samples with synonyms

        Parameters
        ----------
        sample : str
            Input Text
        """
        words = set(word_tokenize(sample))
        chosen_words = random.sample(words, self.number_of_replacements)
        
        for chosen_word in chosen_words:
            synonyms = sum(wordnet.synonyms(chosen_word),[])
            if len(synonyms) == 0:
                continue
            replacement = random.choice(synonyms)
            sample = sample.replace(chosen_word, replacement)
        
        return sample