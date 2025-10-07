import torch
import time
from tqdm import tqdm
import csv


def train_model(model, train_dataloader, validation_dataloader, optimizer, lr_scheduler, device, num_epochs, checkpoint_filepath, patience, model_filepath):
    '''
    Trains a model using specified dataloaders and saves checkpoints and the final model.

    Parameters:
        model (torch.nn.Module): The model to be trained.
        train_dataloader (DataLoader): Dataloader for training data.
        validation_dataloader (DataLoader): Dataloader for validation data.
        optimizer (Optimizer): Optimizer used for training.
        lr_scheduler (LRScheduler): Learning rate scheduler.
        device (torch.device): Device to train the model on.
        num_epochs (int): Number of epochs to train the model.
        checkpoint_filepath (str): File path for saving model checkpoints.
        patience (int): Patience for early stopping.
        model_filepath (str): File path for saving the final trained model.
    
    Returns:
        tuple: Tuple containing lists of training and validation losses.
    '''
    # Moving model to device
    model = model.to(device)

    # Keeping track of losses
    training_losses = []
    validation_losses = []

    # To track the time it takes to complete training
    total_training_start_time = time.time()

    # Best validation loss initialized to a very large number
    best_validation_loss = float('inf')

    # Going through each epoch, Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'labels']}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            training_losses.append(loss.item())

        model.eval()
        validation_loss = 0
        with torch.no_grad():
            for batch in tqdm(validation_dataloader):
                batch = {k: v.to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'labels']}
                outputs = model(**batch)
                validation_loss += outputs.loss.item()

                validation_losses.append(outputs.loss.item())

        validation_epoch_loss = validation_loss / len(validation_dataloader)
        train_epoch_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch}: Training Loss = {train_epoch_loss}, Validation Loss = {validation_epoch_loss}")

        # Extracting the current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr}")

        # Save the model checkpoint if it has the best validation loss so far
        if validation_epoch_loss < best_validation_loss:
            best_validation_loss = validation_epoch_loss

            checkpoint = {
                'epoch': epoch,
                'learning_rate': current_lr,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'training_loss': train_epoch_loss,
                'validation_loss': validation_epoch_loss
            }

            checkpoint_filename = f"{checkpoint_filepath}_epoch_{epoch}.pth"

            # Saving checkpoint
            torch.save(checkpoint, checkpoint_filename)

            # Reset patience counter
            patience_counter = 0

        else:

            # Increment patience counter
            patience_counter += 1

        # Check if patience has run out
        if patience_counter >= patience:
            print(f"Early stopping triggered. Stopping training at epoch {epoch}.")
            break

    total_training_time = time.time() - total_training_start_time
    print("Total Training Time: ", total_training_time)

    # Saving the Finetuned Model
    model.save_pretrained(model_filepath)
    print("Model saved at: ", model_filepath)

    return training_losses, validation_losses

def save_losses(training_losses, validation_losses, file_path):
    '''
    Saves training and validation losses to a CSV file.

    Args:
        training_losses (list): A list of training loss values.
        validation_losses (list): A list of validation loss values.
        file_path (str): The path to the file where the losses will be saved.
    '''
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Training Loss', 'Validation Loss'])

        for train_loss, validation_loss in zip(training_losses, validation_losses):
            writer.writerow([train_loss, validation_loss])
