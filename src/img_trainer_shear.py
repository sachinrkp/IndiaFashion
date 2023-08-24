"""
    Script File to train for Cloth Classification task
"""
import os
import torch
import argparse
import torch.utils.data
from tqdm import tqdm
import torch.optim as optim
import utils.config as config
from model_archs.img_models import ResNetFeaturesFlatten
from utils import tflogger
from utils.common_utils import image_transform, image_transform_shear, get_accuracy, calculate_metrics
from utils.dataset import EthnicFinderDataset
import numpy as np
from utils.common_utils import CLOTH_CATEGORIES
import torch.nn as nn
import torchvision.transforms as transforms


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-m', '--mode', type=str, default='test',
                    help="mode, {'" + "train" + "', '" + "test" + "'}")
parser.add_argument('-n', '--model_name', type=str,
                    default='my_model',
                    help='Model Name')
parser.add_argument('-a', '--architecture', type=str, default='resnet18',
                    help="model architecture, {'" + "', '".join(config.MODELS.keys()) + "'}")

args = parser.parse_args()

model_name = args.model_name

# Dataloaders
train_set = EthnicFinderDataset(metadata_file=config.train_file, mode='train', transform=image_transform_shear)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=2)
val_set = EthnicFinderDataset(metadata_file=config.val_file, mode='val', transform=image_transform)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=2)
test_set = EthnicFinderDataset(metadata_file=config.test_file, mode='test', transform=image_transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=config.batch_size, shuffle=True, num_workers=2)


# Model
cloth_model = ResNetFeaturesFlatten(model_key=args.architecture).to(config.device)

# Optimizer
optimizer = optim.Adam(cloth_model.parameters(), lr=config.lr)
print("Total Params", sum(p.numel() for p in cloth_model.parameters() if p.requires_grad))

# Logger
logger = tflogger.Logger(model_name=model_name, data_name='ours',
                         log_path=os.path.join(config.BASE_DIR, 'tf_logs', model_name))



def compute_class_weights(metadata_list):
    class_counts = {class_label: 0 for class_label in CLOTH_CATEGORIES.keys()}  # Initialize counts for all class labels

    for item in metadata_list:
        class_label = item["class_label"]
        class_counts[class_label] += 1  # Increment count for the encountered class label

    num_classes = len(CLOTH_CATEGORIES)
    print("Number of classes:", num_classes)
    print("Class counts:", class_counts)

    # Check if all class labels in CLOTH_CATEGORIES are present in the dataset's metadata
    missing_labels = set(CLOTH_CATEGORIES.keys()) - set(class_counts.keys())
    if missing_labels:
        print("Missing class labels in metadata:", missing_labels)

    class_weights = [1.0 / (class_counts[class_label] + 1e-6) for class_label in CLOTH_CATEGORIES.keys()]
    print("Class weights:", class_weights)

    return class_weights


def train_epoch(epoch, class_weights):
    train_loss = 0.
    total = 0.
    correct = 0.
    cloth_model.train()
    # Training loop
    for batch_idx, (image, label) in enumerate(tqdm(train_loader)):
        image = image.to(config.device)
        label = label.to(config.device)

        batch = image.shape[0]
        with torch.set_grad_enabled(True):
            y_pred = cloth_model(image)
            loss = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(config.device))(y_pred, label)
            loss.backward()
            train_loss += float(loss.item())
            optimizer.step()
            optimizer.zero_grad()
            correct += get_accuracy(y_pred, label)
            total += batch

    # Calculate loss and accuracy for the current epoch
    logger.log(mode="train", scalar_value=train_loss / len(train_loader), epoch=epoch, scalar_name='loss')
    logger.log(mode="train", scalar_value=correct / total, epoch=epoch, scalar_name='accuracy')

    print(' Train Epoch: {} Loss: {:.4f} Acc: {:.2f} '.format(epoch, train_loss / len(train_loader), correct / total))


def eval_epoch(epoch):
    cloth_model.eval()
    val_loss = 0.
    total = 0.
    correct = 0.
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch_idx, (image, label) in enumerate(tqdm(val_loader, desc='')):
            image = image.to(config.device)
            label = label.to(config.device)
            batch = image.shape[0]
            y_output = cloth_model(image)
            loss = config.ce_criterion(y_output, label)
            val_loss += float(loss.item())
            predicted_labels = torch.argmax(y_output, dim=1)
            correct += torch.sum(predicted_labels == label).item()
            total += batch
            y_true.extend(label.tolist())
            y_pred.extend(predicted_labels.tolist())

    precision, recall, f1_score, error_rate = calculate_metrics(y_true, y_pred)
    error_rate = 1.0 - (correct / total)

    logger.log(mode="val", scalar_value=val_loss / len(val_loader), epoch=epoch, scalar_name='loss')
    logger.log(mode="val", scalar_value=correct / total, epoch=epoch, scalar_name='accuracy')
    logger.log(mode="val", scalar_value=precision, epoch=epoch, scalar_name='precision')
    logger.log(mode="val", scalar_value=recall, epoch=epoch, scalar_name='recall')
    logger.log(mode="val", scalar_value=f1_score, epoch=epoch, scalar_name='f1_score')
    logger.log(mode="val", scalar_value=error_rate, epoch=epoch, scalar_name='error_rate')

    print(' Val Epoch: {} Avg loss: {:.4f} Acc: {:.2f} Precision: {:.4f} Recall: {:.4f} F1-Score: {:.4f} Error Rate: {:.4f}'.format(
        epoch, val_loss / len(val_loader), correct / total, precision, recall, f1_score, error_rate))
    return val_loss


def train_model():
    try:
        print("Loading Saved Model")
        checkpoint = torch.load(config.BASE_DIR + 'models/' + model_name + '.pt')
        cloth_model.load_state_dict(checkpoint)
        print("Saved Model successfully loaded")
        cloth_model.eval()
        best_loss = eval_classifier_full()
    except:
        best_loss = np.Inf  # Initialize the best_loss with a large value

    early_stop = False
    counter = 0
    # Calculate class weights for the loss function
    class_weights = compute_class_weights(train_set.metadata_list)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(config.device)

    for epoch in range(1, config.epochs + 1):
        # Training epoch
        train_epoch(epoch, class_weights)
        # Validation epoch
        avg_test_loss = eval_epoch(epoch)
        if avg_test_loss <= best_loss:
            counter = 0
            best_loss = avg_test_loss
            torch.save(cloth_model.state_dict(), config.BASE_DIR + 'models/' + model_name + '.pt')
            print("Best model saved/updated..")
        else:
            counter += 1
            if counter >= config.patience:
                early_stop = True
        # If early stopping flag is true, then stop the training
        if early_stop:
            print("Early stopping")
            break


def eval_classifier_full():
    val_loss = 0.
    total = 0.
    with torch.no_grad():
        for batch_idx, (image, label) in enumerate(tqdm(val_loader, desc='')):
            image = image.to(config.device)
            label = label.to(config.device)
            y_output = cloth_model(image)
            loss = config.ce_criterion(y_output, label)
            val_loss += loss.item()
            total += 1  # Increment total for each batch processed

    avg_val_loss = val_loss / total  # Calculate the average evaluation loss
    print(' Val Avg loss: {:.4f}'.format(avg_val_loss))
    return avg_val_loss



def test_classifier():
    try:
        print("Loading Saved Model")
        checkpoint_path = os.path.join(config.BASE_DIR + 'models/' + model_name + '.pt')
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=config.device)
            cloth_model.load_state_dict(checkpoint)
            print("Saved Model successfully loaded")
            cloth_model.eval()
        else:
            print("Model Not Found")
            exit()
    except:
        print("Error in loading the model")
        exit()
    total = 0.
    correct = 0.
    labels = torch.LongTensor([])
    y_preds = torch.LongTensor([])

    with torch.no_grad():
        for batch_idx, (image, label) in enumerate(tqdm(test_loader, desc='')):
            image = image.to(config.device)
            label = label.to(config.device)
            batch = image.shape[0]
            y_pred = cloth_model(image)
            _, y_pred_cls = torch.max(y_pred, 1)
            labels = torch.cat([labels, label.cpu()])
            y_preds = torch.cat([y_preds, y_pred_cls.cpu()])
            correct += get_accuracy(y_pred, label)
            total += batch

    precision, recall, f1_score, error_rate = calculate_metrics(labels.tolist(), y_preds.tolist())
    error_rate = 1.0 - (correct / total)
    print('Test Acc: {:.4f}'.format(correct / total))
    print('Precision: {:.4f} Recall: {:.4f} F1-Score: {:.4f} Error Rate: {:.4f}'.format(precision, recall, f1_score, error_rate))


if __name__ == '__main__':

    if args.mode == "train":
        train_model()
    elif args.mode == "test":
        test_classifier()
