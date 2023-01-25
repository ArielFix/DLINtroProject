from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import random

class SwapWords:
    def __init__(self, number_of_swaps=5):
        """Augmentation for NLP based models, swaps random words

        Parameters
        ----------
        number_of_replacements : int, optional
            number of words to replace per sentence, by default 5
        """
        self.number_of_swaps = number_of_swaps
    def __call__(self, sample: str) -> str:
        """Swaps location of random words

        Parameters
        ----------
        sample : str
            Input Text
        """
        words = list(word_tokenize(sample))

        for _ in range(self.number_of_swaps):
            i,j = random.sample(range(len(words)), k=2)
            temp = words[i]
            words[i] = words[j]
            words[j] = temp        
        return TreebankWordDetokenizer().detokenize(words)