from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import random

stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our', 
			'ours', 'ourselves', 'you', 'your', 'yours', 
			'yourself', 'yourselves', 'he', 'him', 'his', 
			'himself', 'she', 'her', 'hers', 'herself', 
			'it', 'its', 'itself', 'they', 'them', 'their', 
			'theirs', 'themselves', 'what', 'which', 'who', 
			'whom', 'this', 'that', 'these', 'those', 'am', 
			'is', 'are', 'was', 'were', 'be', 'been', 'being', 
			'have', 'has', 'had', 'having', 'do', 'does', 'did',
			'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
			'because', 'as', 'until', 'while', 'of', 'at', 
			'by', 'for', 'with', 'about', 'against', 'between',
			'into', 'through', 'during', 'before', 'after', 
			'above', 'below', 'to', 'from', 'up', 'down', 'in',
			'out', 'on', 'off', 'over', 'under', 'again', 
			'further', 'then', 'once', 'here', 'there', 'when', 
			'where', 'why', 'how', 'all', 'any', 'both', 'each', 
			'few', 'more', 'most', 'other', 'some', 'such', 'no', 
			'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 
			'very', 's', 't', 'can', 'will', 'just', 'don', 
			'should', 'now', ''])

class RandomInsertion:
    def __init__(self, number_of_insertions=5):
        """Augmentation for NLP based models, insert synonym random places

        Parameters
        ----------
        number_of_replacements : int, optional
            number of words to replace per sentence, by default 5
        """
        self.number_of_insertions = number_of_insertions
    def __call__(self, sample: str) -> str:
        """Replace words in samples with synonyms

        Parameters
        ----------
        sample : str
            Input Text
        """
        words = list(word_tokenize(sample))
        chosen_words = random.sample(words, min(self.number_of_insertions, len(words)))

        for chosen_word in chosen_words:
            if chosen_word in stop_words:
                continue
            synonyms = sum(wordnet.synonyms(chosen_word),[])
            if len(synonyms) == 0:
                continue
            insertion = random.choice(synonyms)
            insertion_idx=random.choice(range(len(words)+1))
            words.insert(insertion_idx, insertion)
        return TreebankWordDetokenizer().detokenize(words)

