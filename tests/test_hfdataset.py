import pytorch_lightning as pl
import torch
import datasets

from typing import Union, Dict

from quaterion_models.encoders import Encoder
from quaterion_models.heads import EncoderHead 
from quaterion_models.heads.skip_connection_head import SkipConnectionHead 
from quaterion import Quaterion, TrainableModel
from quaterion.dataset import GroupSimilarityDataLoader, SimilarityGroupSample
from quaterion.dataset.hf_similarity_dataset import HFSimilarityGroupDataset
from quaterion.loss import OnlineContrastiveLoss, SimilarityLoss
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, Pooling

from torch import Tensor, nn

from quaterion_models.types import TensorInterchange, CollateFnType
from quaterion_models.encoders import Encoder


class IMDBEncoder(Encoder):
    def save(self, output_path: str):
        pass

    @classmethod
    def load(cls, input_path: str) -> "Encoder":
        return IMDBEncoder()

    def __init__(self, transformer, pooling):
        super().__init__()
        self.transformer = transformer
        self.pooling = pooling
        self.encoder = nn.Sequential(self.transformer, self.pooling)

    def get_collate_fn(self) -> CollateFnType:
        return self.transformer.tokenize
    
    @property
    def embedding_size(self) -> int:
        return self.transformer.get_word_embedding_dimension()
    
    @property
    def trainable(self) -> bool:
        return True

    def forward(self, inputs):
        return self.encoder.forward(inputs)["sentence_embedding"]

class Model(TrainableModel):
    def __init__(self, embedding_size: int, lr: float):
        self._embedding_size = embedding_size
        self._lr = lr
        super().__init__()

    def configure_encoders(self) -> Union[Encoder, Dict[str, Encoder]]:
        pre_trained_model = SentenceTransformer("all-MiniLM-L6-v2")
        transformer: Transformer = pre_trained_model[0]
        pooling: Pooling = pre_trained_model[1]
        encoder = IMDBEncoder(transformer, pooling)
        return encoder

    def configure_head(self, input_embedding_size) -> EncoderHead:
        return SkipConnectionHead(input_embedding_size)

    def configure_loss(self) -> SimilarityLoss:
        return OnlineContrastiveLoss()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), self._lr)
        return optimizer

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
        print(type(dataset[0]), type(dataset[0].obj), type(dataset[0].group))

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

    def test_training_model(self):
        dataset = HFSimilarityGroupDataset("imdb", split="train")
        dataset.load_dataset()
        dataset.subset(4, 24)
        print(len(dataset))
        dataloader = GroupSimilarityDataLoader(dataset, batch_size=4)
        model = Model(embedding_size=768, lr=1e-3)
        trainer = pl.Trainer(logger=False, max_epochs=1)

        Quaterion.fit(
            trainable_model=model,
            trainer=trainer,
            train_dataloader=dataloader
        )