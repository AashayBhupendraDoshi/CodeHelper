import os
import openai
import numpy as np
import pandas as pd
from indexing import indexing



class ProcessCode:
    def __init__(self, ignoreFiles: list) -> None:
        openai.api_key = "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"
        self.inference_dir = "./process_files/"
        self.indexer = indexing()
        self.prompt = '''Explain this code to me:
        '''
        self.dirIgnore = set(ignoreFiles)
        pass

    def get_files_in_directory(self,directory):
        file_paths = []
        
        for root, dirs, files in os.walk(directory):
            # Exclude directories starting with a "."
            dirs[:] = [d for d in dirs if ((not d.startswith('.')) and (not d.startswith('_'))) and d not in self.dirIgnore]
            for file in files:
                # Exclude files starting with a "."
                if not file.startswith('.'):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, directory)
                    file_paths.append(relative_path)
        
        return file_paths

    def processRepo(self, root_path: str):
        fileName = []
        summary = []
        embeddings = []

        files_list = self.get_files_in_directory(root_path)
        for i in files_list:
            f = open(root_path + i, "r")
            file_text = f.read() 
        
            completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": self.prompt+file_text}
            ]
            )
            code_summary = completion.choices[0].message["content"]
            indexResponse = self.indexer.indexText(code_summary)
            for j in indexResponse:
                fileName+=[i]
                summary+=[j[0]]
                embeddings+=[j[1]]
        
        output = pd.DataFrame()
        output['fileName'] = fileName
        output['summary'] = summary
        output['embeddings'] = embeddings
        output.to_pickle(self.inference_dir + os.path.split(os.path.split(root_path)[0])[-1]+".pkl")
        return


a = ProcessCode()
b = a.processRepo("/home/abd/Desktop/projects/mumbaiHacks/", ["process_files"])
print(b)



