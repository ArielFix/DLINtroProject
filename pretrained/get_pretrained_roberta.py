from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

class GetPretrainedRoberta:
    def __init__(self):
        """"Initialize new pre-trained roberta model"""

        task='sentiment'
        MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL)
        self.model.save_pretrained(MODEL)


   


        
