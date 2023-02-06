from translators.server import google
import random
from enum import Enum

class Languages(Enum):
    ENGLISH="en"
    SPANISH="es"
    FRENCH="fr"
    GERMAN="de"

    @classmethod
    def get_list(cls):
        return set(cls)

class BackTranslate:
    def __init__(self, inp_language=Languages.ENGLISH, languages=None) -> None:
        """Augmentation for NLP models, transfer from the text's language and back in order to get "close" variations

        Parameters
        ----------
        inp_language : Literal, optional
            Language of the input text, by default Languages.ENGLISH
        languages : Available translation languages, optional
            list of languages, by default None
        """
        self.inp_language = inp_language
        self.languages = tuple(Languages.get_list()) if languages is None else languages

    def __call__(self, sample):
        """Translate sample to and from a random language from languages

        Parameters
        ----------
        sample
            input sample text in input language
        """
        inp_language = self.inp_language.value
        augmentation_language = random.choice(self.languages).value
        if augmentation_language == self.inp_language.value:
            return sample
        sample = google(google(sample, inp_language, augmentation_language), augmentation_language, inp_language)
        return sample
    
    