import pandas as pd



class Utilities:
    def preprocess_text_line(text):

        """
        pre-processing a string:
        
        input:  text (string) 
        output text (strung) 
        """

        new_text = []
 
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)

        return " ".join(new_text)

    def preprocess_text_series(text):
        """
        pre-processing on a given list of string - performed in place:
        
        input:  text (list(string)) 
        """

        for line in range(0, len(text)):
            new_text = []
            twit = text[line]

            for t in twit.split(" "):
                t = '@user' if t.startswith('@') and len(t) > 1 else t
                t = 'http' if t.startswith('http') else t

                new_text.append(t)

            text[line] = " ".join(new_text)

