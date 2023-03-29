import pytorch_lightning as pl
import torch

from quaterion_models.encoders import Encoder
from quaterion_models.heads import EncoderHead, GatedHead
from quaterion import Quaterion, TrainableModel
from quaterion.dataset import GroupSimilarityDataLoader, SimilarityGroupSample
from quaterion.dataset.hf_similarity_dataset import HFSimilarityGroupDataset
from quaterion.loss import OnlineContrastiveLoss, SimilarityLoss

import datasets


class TestHFSimilarityDataset:
    def test_loading_hf_dataset(self):
        hfdataset = datasets.load_dataset("imdb", split="train")
        dataset = HFSimilarityGroupDataset("imdb", split="train")
        dataset.load_dataset()
        assert(dataset[2].obj == hfdataset[2]["text"])
        assert(dataset[2].group == hfdataset[2]["label"])
        assert(type(dataset[2]) == SimilarityGroupSample)
        assert(type(dataset[2].obj == str))
        assert(type(dataset[2].group == int))
        print(type(dataset[2]), type(dataset[2].obj), type(dataset[2].group))

    def test_loading_hf_datasets_multi_feature(self):
        # For datasets where the features are not neatly just record and label
        # Quaterion takes datasets in the form of record and label (obj and group) attrs
        # to make it's SimilarityGroupSample objects. But HF Hub has lots of datasets
        # What happens when these datasets have many features and we can't readily get record
        # and label?
        # Demonstrating the functionality of our useful set_similarity_group_sample_attrs

        hfdataset = datasets.load_dataset("anli", split="train_r1")
        dataset = HFSimilarityGroupDataset("anli", split="train_r1")
        dataset.load_dataset()
        dataset.set_similarity_group_sample_attrs("premise", "label")
        
        assert(dataset[2].obj == hfdataset[2]["premise"])
        assert(dataset[2].group == hfdataset[2]["label"])
        assert(type(dataset[2]) == SimilarityGroupSample)
        assert(type(dataset[2].obj == str))
        assert(type(dataset[2].group == int))
        
        print(type(dataset[2]), type(dataset[2].obj), type(dataset[2].group))

