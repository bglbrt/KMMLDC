
import numpy as np

class StratifiedKFold():


    def __init__(self, n_splits=5):
        self.n_splits = n_splits
        
    def split(self, Xtr, Ytr):
        n_samples = Xtr.shape[0]
        wrapper = np.array([(i, Ytr[i]) for i in range(n_samples)])
        folds = [[] for _ in range(self.n_splits)]
        for y in np.unique(Ytr):
            subset = list(wrapper[np.where(wrapper[:,1] == y)][:,0])
            k, m = divmod(len(subset), self.n_splits)
            for i in range(self.n_splits):
                folds[i] += subset[i*k+min(i, m):(i+1)*k+min(i+1, m)]
        return [([x for j in range(self.n_splits) for x in folds[j] if j != i], folds[i]) for i in range(self.n_splits)]