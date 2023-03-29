from typing import Sized, Union

from torch.utils.data import Dataset

from quaterion.dataset.similarity_samples import SimilarityGroupSample
from quaterion.dataset.similarity_dataset import SimilarityGroupDataset

import datasets as hfds

class HFSimilarityGroupDataset(SimilarityGroupDataset):
    def __init__(self, hf_dataset_name: str, **kwargs):
        self.hf_dataset_name = hf_dataset_name
        self.kwargs = kwargs
        self.record_attr = None
        self.label_attr = None

    def load_dataset(self) -> None:
        self._dataset = hfds.load_dataset(self.hf_dataset_name, **self.kwargs)
        
        if "split" in self.kwargs.keys() and self.kwargs["split"] is not None:
            self.is_dataset_dict = False
        else:
            self.is_dataset_dict = True
        
        if "streaming" in self.kwargs.keys() and self.kwargs["streaming"] == True:
            self.iterable_dataset = True
        else:
            self.iterable_dataset = False

    def __len__(self) -> int:
        if isinstance(self._dataset, Sized):
            return len(self._dataset)
        else:
            raise NotImplementedError
        
    def set_similarity_group_sample_attrs(self, record_attr: str, label_attr: str) -> None:
        '''Allows the user to specify by name the features in the dataset
        that are to be used as the record, and the group number.
        This is especially useful as datasets of many different characteristics
        may be pulled from HF hub.
        Allowing the user this flexibility avoids the framework from falling
        into the trap of trying to do too much with too little and instead
        provides the user with a useful interface that will seamlessly adapt
        to a range of scenarios
        '''
        self.record_attr = record_attr
        self.label_attr = label_attr

    def __getitem__(self, index:Union[int, str]) -> SimilarityGroupSample:
        if isinstance(self._dataset, hfds.IterableDataset):
            raise NotImplementedError
        if isinstance(self._dataset, hfds.DatasetDict):
            return self._dataset[index] # returns the 'index' split of the dataset
        if self.record_attr == None and self.label_attr == None:
            item = self._dataset.__getitem__(index)
            if isinstance(item, dict):
                record, label = item.values()
            else:
                record, label = item
        else:
            record = self._dataset[self.record_attr].__getitem__(index)
            label = self._dataset[self.label_attr].__getitem__(index)
        
        return SimilarityGroupSample(obj=record, group=label)
    
    def __iter__(self):
        if self.iterable_dataset == True:
            if self.record_attr == None and self.label_attr == None:
                yield from iter(self._dataset)
            else:
                raise NotImplementedError
                # obj = next(iter(self._dataset))
                # yield obj[self.record_attr], obj[self.label_attr]

        else:
            raise NotImplementedError
