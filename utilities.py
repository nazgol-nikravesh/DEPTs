import numpy as np
import pandas as pd
from scipy.io import arff


##### READING DATASET
def read_dataset(directory, dataset_name):

    if dataset_name in ["ant-1.3", "ant-1.4", "ant-1.5", "ant-1.6", "ant-1.7","camel-1.0","camel-1.2","camel-1.4","camel-1.6","ivy-1.1","ivy-1.4","ivy-2.0"
                        ,"jedit-3.2","jedit-4.0","jedit-4.1","jedit-4.2","jedit-4.3","log4j-1.0","log4j-1.1","log4j-1.2","lucene-2.0","lucene-2.2","lucene-2.4"
                        ,"poi-1.5","poi-2.0","poi-2.5","poi-3.0","synapse-1.0","synapse-1.1","synapse-1.2","velocity-1.4","velocity-1.5","velocity-1.6",
                        "xerces-1.2","xerces-1.3","xerces-1.4","xerces-init"]:
        X = pd.read_csv(directory + dataset_name + '.csv')
        y = X['bug']
        del X['bug']

    elif dataset_name in ["KC3", "PC2", "PC4", "PC5", "JM1", "MC1", "KC1", "PC3" ,"PC1" ,"MW1" ,"CM1" ,"MC2" ]:
        data, meta = arff.loadarff(directory + dataset_name + '.arff')
        X =  pd.DataFrame(data)

        y = X['Defective']
        y = mapit(y)
        del X['Defective']

    else:
        print("dataset %s does not exist" % dataset_name)


    return np.array(X), np.array(y)
#### MISC
def mapit(vector):

    s = np.unique(vector)

    mapping = pd.Series([x[0] for x in enumerate(s)], index = s)
    vector=vector.map(mapping)
    return vector
