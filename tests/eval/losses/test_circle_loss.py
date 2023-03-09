import torch

from quaterion.loss import CircleLoss


class TestCircleLoss:
    embeddings = torch.Tensor(
        [
            [0.0, -1.0, 0.5],
            [0.1, 2.0, 0.5],
            [0.0, 0.3, 0.2],
            [1.0, 0.0, 0.9],
            [1.2, -1.2, 0.01],
            [-0.7, 0.0, 1.5],
        ]
    )

    groups = torch.LongTensor([1, 2, 3, 3, 2, 1])

    def test_batch_all(self):
        loss = CircleLoss(margin=0.5, scale_factor=2)

        loss_res = loss.forward(embeddings=self.embeddings, groups=self.groups)

        assert loss_res.shape == torch.Size([])
