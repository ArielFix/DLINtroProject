from augmentations.deletion import RandomDelete
from augmentations.insertion import RandomInsertion
from augmentations.remove_duplicates import RemoveDuplicates
from augmentations.swap import SwapWords
from augmentations.synonym import SynonymReplace
from augmentations.translation import BackTranslate
import nltk
nltk.download('punkt')
nltk.download('wordnet')


sample_text = "Hello @world, let's see if the #augmentations work! GO! GO! GO!"

print("Synonym Replace:", SynonymReplace()(sample_text))
print("Back Translate:", BackTranslate()(sample_text))
print("Random Delete:", RandomDelete()(sample_text))
print("Random Insertion:", RandomInsertion()(sample_text))
print("Remove Duplicates:", RemoveDuplicates()(sample_text))
print("Swap Words:", SwapWords()(sample_text))

