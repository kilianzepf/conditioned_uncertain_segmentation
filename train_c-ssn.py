import os
from tqdm import tqdm
import argparse
from types import SimpleNamespace
from datetime import datetime

import torch
from torch._C import device
import torch.nn as nn
import torch.optim as optim

from metadata_manager import *
from utils.utils import *
from utils.metrics import *
from models.c_ssn import StyleStochasticUnet
from dataloaders import *


parser = argparse.ArgumentParser()
parser.add_argument(
    "--what",
    default="isic3_style_concat",
    help="Dataset to train on.",
)
parser.add_argument(
    "--lr",
    default=0.0001,
    type=float,
    help="Learning Rate for Training. Default is 0.0001",
)
parser.add_argument(
    "--rank",
    default=10,
    type=int,
    help="Rank for Covoriance decomposition. Default is 10",
)
parser.add_argument(
    "--epochs", default=200, type=int, help="Number of Epochs to train. Default is 200"
)
parser.add_argument(
    "--batchsize", default=6, type=int, help="Number of Samples per Batch. Default is 6"
)
parser.add_argument(
    "--weightdecay",
    default=1e-4,
    type=float,
    help="Parameter for Weight Decay. Default is 1e-4",
)
parser.add_argument(
    "--resume_epoch",
    default=0,
    type=int,
    help="Resume training at the specified epoch. Default is 0",
)
parser.add_argument(
    "--save_model",
    default=False,
    type=bool,
    help="Set True if checkpoints should be saved. Default is False",
)
parser.add_argument(
    "--test_treshold",
    default=0.5,
    type=float,
    help="Treshold for masking the logid/sigmoid predictions. Only use with --testit. Default is 0.5",
)
parser.add_argument(
    "--N", default=16, type=int, help="Number of Samples for GED Metric. Default is 16"
)
parser.add_argument(
    "--W",
    default=1,
    type=int,
    help="Set 0 to turn off Weights and Biases. Default is 1 (tracking)",
)
parser.add_argument(
    "--transfer",
    default="None",
    help="Activates transfer learning when given a model's name. Default is None (no transfer learning)",
)
parser.add_argument(
    "--num_filters",
    default=[32, 64, 128, 192],
    nargs="+",
    help="Number of filters per layer. Default is [32,64,128,192]",
    type=int,
)


def train(
    model,
    resume_epoch,
    epochs,
    opt,
    train_loader,
    val_loader,
    save_checkpoints,
    transfer_model,
    metadata,
    forward_passes,
    W=True,
):
    # Set device to Cuda if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Check if want to resume prior checkpoints
    if resume_epoch > 0:
        print(f"Resuming training on epoch {resume_epoch} ... \n")
        # Load Checkpoint
        checkpoint = torch.load(
            f"checkpoints/{meta.directory_name}/{model.name}/{resume_epoch}_checkpoint.pt"
        )
        # Inject checkpoint to model and optimizer
        model.load_state_dict(checkpoint["model_state_dict"])
        opt.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]
    if transfer_model != "None":
        print(f"Continue Training on the model {transfer_model}...\n")
        transfer_dict = torch.load(
            f"saved_models/{meta.directory_name}/{transfer_model}.pt"
        )
        model = transfer_dict["model"]
        opt = transfer_dict["optimizer"]
        loss = transfer_dict["loss"]
    else:
        print(f"Training from scratch...\n")

    # Log model to wandb
    if W:
        wandb.watch(model, log_freq=100)
    iterations = 0
    for epoch in tqdm(range(resume_epoch, epochs)):  # may be error in range
        sum_batch_loss = 0
        sum_batch_IoU = 0
        counter = 0
        model.train()
        for images, masks, _, style_label in train_loader:
            counter += 1
            iterations += 1
            # Send tensors to Cuda
            images = images.to(device)
            style_label = style_label.to(device)
            masks = masks.to(device)

            # Set parameter gradients to None
            opt.zero_grad()
            # Forward pass
            logits, output_dict, logging_infos_of_that_step = model(
                images, style_label
            )  # outputs logits

            logit_distribution = output_dict["distribution"]
            # Treshold (default 0.5)
            pred_mask = torch.sigmoid(logits).ge(meta.masking_threshold)

            # Calculate Loss
            loss_function = StochasticSegmentationNetworkLossMCIntegral(
                num_mc_samples=20
            )
            loss = loss_function(logits, masks, logit_distribution)
            sum_batch_loss += float(loss)

            # Calculate IoU for this prediction
            batch_IoU = IoU(masks, pred_mask)
            sum_batch_IoU += float(batch_IoU)
            # Backward pass & weight update
            loss.backward()
            opt.step()

            # Logging
            mean = logging_infos_of_that_step["mean"].detach().cpu()
            cov_factor = logging_infos_of_that_step["cov_factor"].detach().cpu()
            cov_diag = logging_infos_of_that_step["cov_diag"].detach().cpu()
            logging_infos_of_that_step.pop("mean")
            logging_infos_of_that_step.pop("cov_factor")
            logging_infos_of_that_step.pop("cov_diag")
            # wandb.log({"Estimated Mean Vector of Batch" : wandb.Histogram(mean)}, step=iterations)
            # wandb.log({"Estimated cov_factor Vector of Batch" : wandb.Histogram(cov_factor)}, step=iterations)
            # wandb.log({"Estimated cov_diag Vector of Batch" : wandb.Histogram(cov_diag)}, step=iterations)
            # wandb.log(logging_infos_of_that_step, step=iterations)

            # Log images, targets and predictions to wandb every fifty epochs
            if (epoch % 50 == 0) and (counter == 2):
                grid = make_image_grid(
                    images, masks, torch.sigmoid(logits), required_padding=(0, 0, 0, 0)
                )
                if W:
                    wandb.log(
                        {
                            "Images during Training": [
                                wandb.Image(
                                    grid, caption="Images, Targets, Predictions"
                                )
                            ]
                        },
                        step=iterations,
                    )

        # Log the average loss of all batches in the epoch to Wandb
        if W:
            wandb.log(
                {
                    "Average Loss per Epoch while Training": sum_batch_loss
                    / (len(train_loader)),
                    "Average IoU per Epoch while Training": sum_batch_IoU
                    / (len(train_loader)),
                },
                step=iterations,
            )

        if save_checkpoints == True:
            os.makedirs(
                f"checkpoints/{meta.directory_name}/{model.name}", exist_ok=True
            )
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "loss": loss,
                },
                f"checkpoints/{meta.directory_name}/{model.name}/{epoch+1}_checkpoint.pt",
            )

        # Save the model after last training epoch (for inference or transfer training) to a folder
        if epoch == epochs - 1:
            os.makedirs(f"saved_models/{meta.directory_name}", exist_ok=True)
            torch.save(
                {"model": model, "optimizer": opt, "loss": loss},
                f"saved_models/{meta.directory_name}/{model.name}.pt",
            )

        """
        Evaluate on the validation set and track to see if overfitting happens
        """

        sum_IoU = 0
        sum_loss = 0
        counter = 0
        model.eval()
        with torch.no_grad():
            for images, masks, seg_dist, style_label in val_loader:
                counter += 1
                # Send tensors to cuda
                images = images.to(device)
                masks = masks.to(device)
                style_label = style_label.to(device)
                seg_dist = [x.to(device) for x in seg_dist]

                # IoU/Loss on Image Level
                logits, output_dict, _ = model(images, style_label)
                logit_distribution = output_dict["distribution"]
                pred_mask = (torch.sigmoid(logits)).ge(meta.masking_threshold)
                # Calculate the average ratio between foreground and background pixels
                loss_function = nn.BCEWithLogitsLoss()
                sum_IoU += IoU(masks, pred_mask)
                sum_loss += loss_function(logits, masks)

                # Log images, targets and predictions of the first batch to wandb every 50 epochs
                if (epoch % 50 == 0) and (counter == 2):
                    grid = make_image_grid(
                        images,
                        masks,
                        torch.sigmoid(logits),
                        required_padding=(0, 0, 0, 0),
                    )
                    if W:
                        wandb.log(
                            {
                                "Images during Validation": [
                                    wandb.Image(
                                        grid, caption="Images, Targets, Predictions"
                                    )
                                ]
                            },
                            step=iterations,
                        )

                # Log mean and variance of logit distribution for fixed sample in validation set every 10 epochs
                if (epoch % 10 == 0) and (counter == 2):
                    mean = logit_distribution.mean
                    variance = logit_distribution.variance
                    grid1 = torchvision.utils.make_grid(mean, len(mean))
                    grid2 = torchvision.utils.make_grid(variance, len(variance))
                    if W:
                        wandb.log(
                            {
                                "Mean of Logit Distribution on fixed batch of validation set": [
                                    wandb.Image(grid1, caption="Mean")
                                ]
                            },
                            step=iterations,
                        )
                        wandb.log(
                            {
                                "Variance of Logit Distribution on fixed batch of validation set": [
                                    wandb.Image(grid2, caption="Variance")
                                ]
                            },
                            step=iterations,
                        )

        if W:
            wandb.log(
                {
                    "Loss on Validation Set (post Epoch)": sum_loss / len(val_loader),
                    "IoU on Validation Set (post Epoch)": sum_IoU / len(val_loader),
                },
                step=iterations,
            )


if __name__ == "__main__":
    # Load parsed arguments from command lind
    args = parser.parse_args()

    what_task = args.what
    resume_epoch = args.resume_epoch
    epochs = args.epochs
    batch_size = args.batchsize
    learning_rate = args.lr
    weight_decay = args.weightdecay
    save_checkpoints = args.save_model
    forward_passes = args.N
    rank = args.rank
    W = bool(args.W)  # Bool for turning off wandb tracking
    transfer_model = args.transfer
    num_filters = args.num_filters

    if W:
        import wandb

        wandb.login()
    # Read in Metadata for the task chosen in command line
    meta_dict = get_meta(what_task)
    meta = SimpleNamespace(**meta_dict)

    # Hand some information about the current run to Wandb Panel
    config = dict(
        epochs=epochs,
        resumed_at=resume_epoch,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        loss="See Paper",
        architecture="Conditioned SSN",
        dataset=meta.description,
        N_for_metrics=forward_passes,
        rank=rank,
        filter=num_filters,
    )

    if W:
        # Create the Training Run in Wandb
        wandb.init(
            project="labelstyle_iclr23",
            group="SSN U-nets",
            job_type="Training",
            config=config,
            dir="/scratch/kmze",
        )
        training_run_name = wandb.run.name
    else:
        # Use current timestamp as name e.g. 2021_12_11_14_46
        training_run_name = (
            str(datetime.now())[:16]
            .replace(" ", "_")
            .replace("-", "_")
            .replace(":", "_")
        )

    print(f"Modelname: {training_run_name}")
    # Check for GPU
    if torch.cuda.is_available():
        print("\nThe model will be run on GPU.")
    else:
        print("\nNo GPU available!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing the {meta.description} dataset.\n")

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if device == "cuda":
        torch.cuda.manual_seed(230)

    # Init a model
    ssn = StyleStochasticUnet(
        name=training_run_name,
        num_channels=meta.channels,
        rank=rank,
        num_filters=num_filters,
        diagonal=False,
    ).to(device)

    # Count number of total parameters in the model and log
    pytorch_total_params = sum(p.numel() for p in ssn.parameters())
    if W:
        wandb.run.summary["Total Model Parameters"] = pytorch_total_params

    # Note that Weight Decay and L2 Regularization are not the same (except for SGD) see paper: Hutter 2019 'Decoupled Weight Decay Regularization'
    # AdamW implements the correct weight decay as shown in their paper
    opt = optim.AdamW(ssn.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Fetch Dataloaders
    train_loader, _ = get_dataloader(
        task=what_task, split="train", batch_size=batch_size, shuffle=True
    )
    val_loader, _ = get_dataloader(
        task=what_task, split="val", batch_size=4, shuffle=False
    )

    # Empty GPU Cache
    torch.cuda.empty_cache()
    # Start Training
    train(
        ssn,
        resume_epoch,
        epochs,
        opt,
        train_loader,
        val_loader,
        save_checkpoints,
        transfer_model,
        meta,
        forward_passes,
        W=W,
    )

    print(f"Saved: {training_run_name} Data: {what_task} Model: SSN")
    # End Training Run
    if W:
        wandb.finish
