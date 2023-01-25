from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer


class RemoveDuplicates:
    def __init__(self, ):
        """Augmentation for NLP based models, remove duplicates

        Parameters
        ----------
        number_of_replacements : int, optional
            number of words to replace per sentence, by default 5
        """
    def __call__(self, sample: str) -> str:
        """Replace words in samples with synonyms

        Parameters
        ----------
        sample : str
            Input Text
        """
        words = word_tokenize(sample)
        out_words = set()
        out_sample = []
        for word in words:
            if word in out_words:
                continue
            out_words.add(word)
            out_sample.append(word)
        return TreebankWordDetokenizer().detokenize(out_sample)

