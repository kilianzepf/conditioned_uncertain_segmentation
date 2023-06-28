import torch.nn as nn
import torch
import torch.distributions as td
import numpy as np

from utils.utils import *
from models.unet import Unet


class StyleStochasticUnet(nn.Module):
    def __init__(
        self,
        name,
        num_channels=1,
        num_classes=1,
        num_filters=[32, 64, 128, 192],
        rank: int = 10,
        epsilon=1e-5,
        diagonal=False,
    ):
        super().__init__()
        self.name = name
        self.rank = rank
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.epsilon = epsilon
        conv_fn = nn.Conv2d
        # whether to use only the diagonal (independent normals)
        self.diagonal = diagonal
        self.mean_l = conv_fn(num_filters[0], num_classes, kernel_size=1)
        self.log_cov_diag_l = conv_fn(num_filters[0], num_classes, kernel_size=1)
        self.cov_factor_l = conv_fn(num_filters[0], num_classes * rank, kernel_size=1)

        self.style_encoder = conv_fn(3, num_filters[0], kernel_size=1)
        self.end_encoder = conv_fn(num_filters[0] * 2, num_filters[0], kernel_size=1)

        self.unet = Unet(
            name=self.name,
            input_channels=self.num_channels,
            num_classes=self.num_classes,
            num_filters=[32, 64, 128, 192],
            apply_last_layer=False,
        )

    def tile(self, a, dim, n_tile):
        """
        This function is taken form PyTorch forum and mimics the behavior of tf.tile.
        Source: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
        Tile means Fliese in Deutsch
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(
            np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])
        ).to(device)

        return torch.index_select(a, dim, order_index)

    def forward(self, image, style):
        logits = self.unet.forward(image)
        batch_size = logits.shape[0]  # Get the batchsize

        # Concatenate Feature Map from U-Net with the tiled Style Variable
        # Style has dimension batchsize x 1
        # Convert to one-hot vectors of size batchsize x 3
        style = torch.nn.functional.one_hot(style, num_classes=3)

        # Tile the conditional style variable to image size batchsizex3xHxW
        style = torch.unsqueeze(style, 2)  # give the tensor another dimension
        style = self.tile(style, 2, logits.shape[2])
        style = torch.unsqueeze(style, 3)  # give the tensor another dimension
        style = self.tile(style, 3, logits.shape[3])
        style = style.float()
        style = self.style_encoder(style)
        # Concatenate the tiled conditional style variable fed through the encoder layer  with the feature map from the U-net along the channel axis
        logits = torch.cat((logits, style), dim=1)
        # Feed the concatenated feature map through the end encoder layer
        logits = self.end_encoder(logits)

        # tensor size num_classesxHxW
        event_shape = (self.num_classes,) + logits.shape[2:]

        mean = self.mean_l(logits)
        cov_diag = self.log_cov_diag_l(logits).exp() + self.epsilon
        # Flattens out each image in the batch, size is batchsize x (rest)
        mean = mean.view((batch_size, -1))
        cov_diag = cov_diag.view((batch_size, -1))

        cov_factor = self.cov_factor_l(logits)
        cov_factor = cov_factor.view((batch_size, self.rank, self.num_classes, -1))
        cov_factor = cov_factor.flatten(2, 3)
        cov_factor = cov_factor.transpose(1, 2)

        cov_factor = cov_factor
        cov_diag = cov_diag + self.epsilon

        # A dictionary that is handed over to the training loop for logging
        infos_for_logging = {
            "mean": mean,
            "cov_factor": cov_factor,
            "cov_diag": cov_diag,
            "Max value of mean": torch.max(mean),
            "Min value of mean": torch.min(mean),
            "Max Value of Cov_diag": torch.max(cov_diag),
            "Max Value of Cov_factor": torch.max(cov_factor),
        }

        if self.diagonal:
            base_distribution = td.Independent(
                td.Normal(loc=mean, scale=torch.sqrt(cov_diag)), 1
            )
        else:
            try:
                base_distribution = td.LowRankMultivariateNormal(
                    loc=mean, cov_factor=cov_factor, cov_diag=cov_diag
                )
            except:
                print(
                    "Covariance became not invertible using independent normals for this batch!"
                )
                base_distribution = td.Independent(
                    td.Normal(loc=mean, scale=torch.sqrt(cov_diag)), 1
                )

        distribution = ReshapedDistribution(
            base_distribution=base_distribution,
            new_event_shape=event_shape,
            validate_args=False,
        )

        shape = (batch_size,) + event_shape
        logit_mean = mean.view(shape)
        cov_diag_view = cov_diag.view(shape).detach()
        cov_factor_view = (
            cov_factor.transpose(2, 1)
            .view((batch_size, self.num_classes * self.rank) + event_shape[1:])
            .detach()
        )

        output_dict = {
            "logit_mean": logit_mean.detach(),
            "cov_diag": cov_diag_view,
            "cov_factor": cov_factor_view,
            "distribution": distribution,
        }

        return logit_mean, output_dict, infos_for_logging
