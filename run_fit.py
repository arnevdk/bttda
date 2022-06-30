from moabb.datasets import BNCI2014008
from moabb.paradigms import P300
from hoda import HODA


paradigm = P300(resample=128, tmin=0.0, tmax=1)
dataset = BNCI2014008()
subject, session, run = 1, "session_0", "run_0"
dataset.download(subject_list=[subject])
data = dataset.get_data(subjects=[subject])
raw = data[subject][session][run]
epochs, labels, _ = paradigm.process_raw(raw, dataset, return_epochs=True)
X = epochs.get_data()
y = labels

hoda = HODA(max_iter=128, rank=(6,6), tol=1e-12, initialize ='random', verbose=True)
hoda.fit(X,y)
