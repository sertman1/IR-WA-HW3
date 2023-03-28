import csv
from typing import Dict, List, NamedTuple

class Document(NamedTuple):
    doc_id: int # NB data files index starting from 1
    sentence: str
    classification_label: int


def get_documents(file):
    docs = []
    with open(file, "r", encoding="utf8") as f:
        tsv_reader = csv.reader(f, delimiter='\t')
        for line in tsv_reader:
            docs.append(Document(int(line[0]), line[2], int(line[1])))
    return docs

def experiment():
    plant_docs = get_documents('./raw_data/plant.tsv') 
    tank_docs = get_documents('./raw_data/tank.tsv')  
    perplace_docs = get_documents('./raw_data/perplace.tsv')   
    smsspam_docs = get_documents('./raw_data/smsspam.tsv')
    
    

    return

if __name__ == '__main__':
    experiment()