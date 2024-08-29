from moabb.datasets import *
from moabb.paradigms import P300,LeftRightImagery
import tensorly as tl

def load_moabb_p300(dataset, subject, session):
    paradigm = P300(resample=48)
    dataset = dataset()
    epochs, labels, meta = paradigm.get_data(
        dataset=dataset, 
         subjects=[subject],
         return_epochs=True
    )
    session = meta['session'][session]
    idc = meta['session'] == session
    epochs = epochs[idc]
    labels = labels[idc]
    meta = meta[idc]
    return epochs, labels, meta

epochs, labels, meta = load_moabb_p300(BNCI2014008,1,1)
X = tl.tensor(epochs.get_data())
y = labels
