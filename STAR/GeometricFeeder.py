import os
import os.path as osp
from time import sleep

import torch
from torch_geometric.data import Dataset, Data
from tqdm import tqdm
from skeleton import process_skeleton, skeleton_parts


class SkeletonDataset(Dataset):
    def __init__(self,
                 root,
                 name,
                 use_motion_vector=True,
                 transform=None,
                 pre_transform=None,
                 benchmark='xsub',
                 sample='train'):
        self.name = name
        self.benchmark = benchmark
        self.training_subjects = {1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38}
        self.training_cameras = {2, 3}
        self.sample = sample

        self.num_joints = 25 if 'ntu' in self.name else 18
        self.skeleton_ = skeleton_parts(num_joints=self.num_joints,
                                        dataset=self.name)
        self.use_motion_vector = use_motion_vector
        self.cached_missing_files = None
        super(SkeletonDataset, self).__init__(root, transform, pre_transform)
        print(self.processed_dir)
        print(self.name)
        path = osp.join(self.processed_dir, '{}.pt'.format(self.name))
        self.data, self.labels = torch.load(path)

    @property
    def missing_skeleton_file_names(self):
        if self.cached_missing_files is not None:
            return self.cached_missing_filesq
        f = open(osp.join(self.root, 'samples_with_missing_skeletons.txt'), 'r')
        lines = f.readlines()
        self.cached_missing_files = set([ln[:-1] for ln in lines])
        f.close()
        return self.cached_missing_files

    @property
    def raw_file_names(self):
        print('raw_file_names')
        return [f for f in os.listdir(self.raw_dir)]

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'kinetics_train')

    @property
    def processed_file_names(self):
        print('processed_file_names')
        return '{}.pt'.format(self.name)

    def download(self):
        pass

    def process(self):
        print('process')
        progress_bar = tqdm(self.raw_file_names)
        skeletons, labels = [], []
        i = 0
        for f in progress_bar:
            if 'ntu' in self.name:
                fl = f[-29:-9]
                if fl in self.missing_skeleton_file_names:
                    # print('Skip file: ', fl)
                    continue

                # action_class = int(fl[fl.find('A') + 1: fl.find('A') + 4])
                subject_id = int(fl[fl.find('P') + 1: fl.find('P') + 4])
                camera_id = int(fl[fl.find('C') + 1: fl.find('C') + 4])

                if self.benchmark == 'cv':
                    if self.sample == 'train':
                        if camera_id not in self.training_cameras:
                            continue
                    else:
                        if camera_id in self.training_cameras:
                            continue

                if self.benchmark == 'cs':
                    if self.sample == 'train':
                        if subject_id not in self.training_subjects:
                            continue
                    else:
                        if subject_id in self.training_subjects:
                            continue
            elif 'kinetics' in self.name:
                import json
                with open(osp.join(self.root,
                                   'kinetics_{}_label.json'.format(self.sample)), 'r') as b:
                    _labels = json.load(b)
            # Read data from `raw_path`.
            sleep(1e-4)
            progress_bar.set_description("processing %s" % f)
            data, label = process_skeleton(osp.join(self.raw_dir,f),
                                           num_joints=self.num_joints,
                                           dataset_name=self.name,
                                           use_motion_vector=self.use_motion_vector)
            if data is None:
                continue

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data = Data(x=data, y=label)  # , edge_index=self.skeleton_)
            skeletons.append(data)
            labels.append(label)
            i += 1

        torch.save([skeletons, torch.FloatTensor(labels)],
                   osp.join(self.processed_dir,
                            self.processed_file_names))

    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]


SkeletonDataset('/Users/zouhuanxin/Downloads/kinetics-skeleton', 'kinetics')