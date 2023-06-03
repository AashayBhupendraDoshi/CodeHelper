import numpy as np
from sentence_transformers import SentenceTransformer


class indexing:
    def __init__(self, chunckSize = 3) -> None:
        self.inputSize = chunckSize
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        pass
    

    def indexText(self,text: str) -> list:
        text = text.split('.')
        output = []
        
        for i in range(0, len(text), self.inputSize):
            output+=[["".join(text[i:i+3]), self.model.encode("".join(text[i:i+3]))]]
        
        return output


    def indexQuery(self, text: str) -> np.ndarray:
        return self.model.encode(text)
    