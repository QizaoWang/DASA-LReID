from __future__ import print_function, absolute_import
import os.path as osp
import re


class Dataset_MSMT(object):
    def __init__(self, root):
        self.root = root
        self.train, self.val, self.trainval = [], [], []
        self.query, self.gallery = [], []
        self.num_train_ids, self.num_val_ids, self.num_trainval_ids = 0, 0, 0

    @property
    def images_dir(self):
        return None

    def load(self, verbose=True):
        exdir = self.root
        self.train, train_pids = self._pluck_msmt(osp.join(exdir, 'list_train.txt'), 'train')
        self.val, val_pids = self._pluck_msmt(osp.join(exdir, 'list_val.txt'), 'train')
        self.train = self.train
        self.query, query_pids = self._pluck_msmt(osp.join(exdir, 'list_query.txt'), 'test')
        self.gallery, gallery_pids = self._pluck_msmt(osp.join(exdir, 'list_gallery.txt'), 'test')
        self.num_train_pids = len(list(set(train_pids)))

        if verbose:
            print("=> ", self.__class__.__name__, "dataset loaded")
            print("  ---------------------------")
            print("  subset   | # ids | # images")
            print("  ---------------------------")
            print("  train    | {:5d} | {:8d}"
                  .format(self.num_train_pids, len(self.train)))
            print("  query    | {:5d} | {:8d}"
                  .format(len(query_pids), len(self.query)))
            print("  gallery  | {:5d} | {:8d}"
                  .format(len(gallery_pids), len(self.gallery)))
            print("  ---------------------------")

    def _pluck_msmt(self, list_file, subdir, pattern=re.compile(r'([-\d]+)_([-\d]+)_([-\d]+)')):
        with open(list_file, 'r') as f:
            lines = f.readlines()
        ret = []
        pids = []
        for line in lines:
            line = line.strip()
            fname = line.split(' ')[0]
            pid, _, cam = map(int, pattern.search(osp.basename(fname)).groups())
            if pid not in pids:
                pids.append(pid)
            ret.append((osp.join(self.root, subdir, fname), pid, cam, 3))
        return ret, pids


class MSMT17(Dataset_MSMT):
    def __init__(self, root):
        super(MSMT17, self).__init__(root)
        self.load()
