from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

class GetPretrainedRoberta:
    def __init__(self):
        """"Initialize new pre-trained roberta model
        initailized object atributes:
        tokenizer: preprocess the strings into a vecor of indexes in the dictionary.
        nodel: the pretrained roberta model for tweet sentoment calssification"""

        task='sentiment'
        MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL)
        # self.model.save_pretrained(MODEL)


   


        
