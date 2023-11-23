import os
import re
import csv
import random
import matplotlib.pyplot as plt
from torchsummary import summary
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
#from pytorchtools import EarlyStopping
import torch.optim as optim
import sklearn.metrics as metrics
from squeeze_and_excitation_model import OAI_KLG_Model

def plot_confMat(target, predicted, out_parentDir="./confusion_matrix_result"):
    labels=["negative", "positive"]
    """
    target of shape: (# of samples,)
    predicted of shape: (# of samples,)
    """
    plt.figure(figsize=(5, 5))
    cm = metrics.confusion_matrix(target, predicted, labels=list(range(2)))
    cm_ratio = cm.astype("float") / cm.sum(axis=1).reshape(-1, 1)
    df_cm = pd.DataFrame(cm_ratio, index=labels, columns=labels)

    sns.heatmap(df_cm, annot=cm, fmt="d", cmap="Blues", vmax=1, vmin=0, annot_kws={"fontsize": 20})
    plt.xlabel("Predicted label", fontsize=18), plt.ylabel("True label", fontsize=18), plt.yticks(rotation=0)
    plt.savefig(f"{out_parentDir}/confusion_matrix_nse_binary"+ str(seed) +".png", bbox_inches="tight", pad_inches=0)
    print(f"You can find the confusion matrix following this path {out_parentDir}/confusion_matrix_se_binary.png")


def custom_transform(image):
    # Separate the two channels

    
    # Apply different modes to each channel
    # pil_channel1 = transforms.ToPILImage(mode='I;16')(channel1)
    # pil_channel2 = transforms.ToPILImage(mode='L')(channel2)

    # PIL does not support creating images with different modes 
    # for each channel directly using Image.fromarray().
    # pil_channel1 = Image.fromarray(channel1.astype(np.uint16), mode="I;16")
    # pil_channel2 = Image.fromarray(channel2.astype(np.uint8), mode="L")


    channel1_array = np.uint8(image.clip(max=255))
    


    

    return torch.tensor(channel1_array) # should be float32

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class CustomDataset(Dataset):
    def __init__(self, dataset, directory, transform=None, is_training = False):

        self.transform = transform
        self.dataset = dataset
        self.directory = directory

        image_sum = 0
        image_squared_sum = 0
        count = 0
        # I assume the image shape is (2, 384, 384), and label shape is (1,). 
        # During training I compute the image mean and standard deviation for both channels
        if (is_training):
            for file in self.dataset:
                data = np.load(self.directory + file)
                image_raw = data['image_raw']
                
                # Convert images to tensors. 
                # TODO how can I convert the first array without loosing information since torch does not support
                # uint16 as data type
                # image_raw = torch.tensor(image_raw, dtype=torch.float32)
                # image_seg = torch.tensor(image_seg, dtype=torch.uint8)         

                
                image_sum += image_raw.mean()
                image_squared_sum += (image_raw * image_raw).mean()
                count += 1
            # shape: (2, 1, 1)
            self.image_mean = image_sum / count 
            image_squared_mean = image_squared_sum / count
            
            # shape: (2, 1, 1)
            image_variance = image_squared_mean - (self.image_mean * self.image_mean) 
            self.image_std = np.sqrt(image_variance)

            # Save these values globally to reuse them to standardize validation and test set
            global global_image_mean
            global_image_mean = self.image_mean
            global global_image_std
            global_image_std = self.image_std
        else:
            self.image_mean = global_image_mean
            self.image_std = global_image_std


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        file = self.dataset[idx]
        data = np.load(self.directory + file)
        image_raw = data['image_raw']
        klg = data['klg']
        label = 0
        if(klg > 1):
            label = 1
        patient_id = file.split("_")[0][4:]
        slice_number = file.split("_")[1][5:8]
        # one_hot_seg_image = []
        # for ar in data['image_seg']:
        #     one_hot_seg_image.append(np.eye(5)[ar])
        # one_hot_seg_image = np.stack(one_hot_seg_image, axis=0)
        # one_hot_seg_image = np.transpose(one_hot_seg_image,(2,0,1))
        # image_raw_tensor = torch.from_numpy(data['image_raw']).unsqueeze(0)
        # one_hot_tensor = torch.from_numpy(one_hot_seg_image)

        # image = torch.cat((image_raw_tensor, one_hot_tensor), dim=0)
        # image = image.float()
        
        # normalization for image raw (grayscale image so maximum is 255 and min is 0)
        # normalization for image segmented (segmented classes so maximum is 4 and min is 0)
        
        image = image_raw.astype(np.float32)

        # apply the transformation if it is defined
        if self.transform:
            return_image = self.transform(image)
        # Standardization
        return_image = (return_image - self.image_mean) / self.image_std

        # converts the image tensor's data type to float
        return_image = return_image.float()

        # return the output
        return return_image, label, torch.tensor(int(slice_number)), patient_id, klg


data_folder = '/home/manfre/workspace/TransUnet/data/Synapse/true_selected_slices/'
folder_path = '/home/manfre/workspace/TransUnet/data/Synapse/true_selected_slices/*.npz'
seed = 54665
random.seed(seed)
# define the global variable for mean and standard deviation.
# these variable will be used for standardization procedure
global_image_mean = 0.0
global_image_std = 0.0
# define the output size
output_size = (224, 224)

# get the list of all files
npz_files = os.listdir(data_folder)


# additional check: filter by files with npz extension
npz_files = [file for file in npz_files if file.endswith('.npz')]

# Extract IDs using regular expressions
id_pattern = re.compile(r'case(\d+)_slice')
ids = [re.search(id_pattern, filename).group(1) for filename in npz_files if re.search(id_pattern, filename)]
unique_ids = list(set(ids))


random.shuffle(unique_ids)

negative_ids = []
positive_ids = []

for id in unique_ids:
    data = np.load(data_folder + f'case{id}_slice010.npz')
    klg_value = data['klg']
    if(klg_value > 1):
        positive_ids.append(id)
    else:
        negative_ids.append(id)

random.shuffle(positive_ids)
random.shuffle(negative_ids)

num_parts = 20
positive_ids_per_part = len(positive_ids) // num_parts # 14
negative_ids_per_part = len(negative_ids) // num_parts # 9

training_ids = positive_ids[:positive_ids_per_part*num_parts - positive_ids_per_part*8] + negative_ids[:negative_ids_per_part*num_parts - negative_ids_per_part*8]
validation_ids = positive_ids[positive_ids_per_part*num_parts - positive_ids_per_part*8: positive_ids_per_part*num_parts - positive_ids_per_part*2] + negative_ids[negative_ids_per_part*num_parts - negative_ids_per_part*8: negative_ids_per_part*num_parts - negative_ids_per_part*2]
testing_ids = positive_ids[positive_ids_per_part*num_parts - positive_ids_per_part*2: ] + negative_ids[negative_ids_per_part*num_parts - negative_ids_per_part*2: ]

random.shuffle(training_ids)
random.shuffle(validation_ids)
random.shuffle(testing_ids)

training_dataset = []
validation_dataset = []
testing_dataset = []

# Add all slices for each ID to the appropriate dataset
for id in training_ids:
    training_dataset.extend([f'case{id}_slice{i:03d}.npz' for i in range(10, 70)] +
                            [f'case{id}_slice{i:03d}.npz' for i in range(90, 150)])

for id in validation_ids:
    validation_dataset.extend([f'case{id}_slice{i:03d}.npz' for i in range(10, 70)] +
                            [f'case{id}_slice{i:03d}.npz' for i in range(90, 150)])

for id in testing_ids:
    testing_dataset.extend([f'case{id}_slice{i:03d}.npz' for i in range(10, 70)] +
                            [f'case{id}_slice{i:03d}.npz' for i in range(90, 150)])



# Initialize a dictionary to store arrays for each ID
id_slices_dict = {}
id_klg_dict = {}

# Iterate through the files and extract 'image_raw'
for filename in npz_files:
    id = filename.split('_')[0][4:]  # Extract the ID from the filename
    data = np.load(data_folder + filename)
    image_raw = data['image_raw']
    label = data['klg']

    if id not in id_slices_dict:
        id_slices_dict[id] = []  # Initialize a list for the ID
    
    id_slices_dict[id].append(image_raw)
    id_klg_dict[id] = int(label)

id_slices_tensors = {}

for id, slices_list in id_slices_dict.items():
    # Convert the list of arrays to a PyTorch tensor
    id_slices_tensors[id] = np.array(slices_list)




# Create a list of arrays (120, x, y) for each ID
# id_arrays_list = [np.stack(id_slices_dict[id]) for id in id_slices_dict]

print("Training dataset:", len(training_dataset), "samples")
print("Validation dataset:", len(validation_dataset), "samples")
print("Testing dataset:", len(testing_dataset), "samples")

training_transform = transforms.Compose([
                                transforms.Lambda(custom_transform),
                                transforms.RandomCrop(output_size), # 224x224
                                ])

validation_transform = transforms.Compose([
                                transforms.Lambda(custom_transform),
                                transforms.CenterCrop(output_size), # 224x224
                                ])


dataset_train = CustomDataset(dataset=training_dataset, directory=data_folder, transform=training_transform, is_training=True)
dataset_val = CustomDataset(dataset=validation_dataset, directory=data_folder, transform=validation_transform)
dataset_test = CustomDataset(dataset=testing_dataset, directory=data_folder, transform=validation_transform)

# create loaders
train_loader = DataLoader(dataset_train, batch_size=256, shuffle=True)
val_loader = DataLoader(dataset_val, batch_size=256, shuffle=False)
test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False)


model = OAI_KLG_Model(in_channel=1, latent_channels=[8, 16, 64, 128, 256, 256], use_slice=true, is_binary_problem = True)
print(sum(p.numel() for p in model.parameters() if p.requires_grad))
print(seed)


# number of parameters  1754698 with latent channels 8, 16, 64, 128, 256, 256 -> 64.42% - 69.40% - 71.91% -> aggregating predictions 70.96% - 75.80% - 82.25% (Simple mean)
# number of parameters  1842058 with latent channels 16, 32, 64, 128, 256, 512 -> 63.13% - 66.76% - 68.13% - 76.37% -> aggregating predictions 70.96% - 70.96% - 79.03%
# number of parameters  2086410 with latent channels 16, 32, 128, 128, 256, 512 -> 65.58% - 66.96% - 69.92% -> aggregating predictions 70.96% - 74.19%
# number of parameters  3070730 with latent channels 16, 32, 128, 256, 256, 512 -> 65.35% 69.65% - 72.31% -> aggregating predictions 72.58% - 77.41% - 83.87% (Simple Mean 25 - 35 epochs)
# number of parameters  7003394 with latent channels 16, 32, 128, 256, 512, 512 -> 65.74% - 67.53% - 72.22% -> aggregating predictions 72.58% - 72.94% - 77.41%

# Remember that the correlated research optimal accuracy value is reached using at most 8181893 parameters

# Define the BCELoss function with weight parameter
weights = torch.Tensor([0.61, 1 - 0.61])  # weight for [0 (negative), 1(positive)]
criterion = torch.nn.BCELoss(reduction="none")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

weights = weights.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
model = model.to(device)

history = {"train":[], "validation":[]}

threshold = 0.5

early_stopper = EarlyStopper(patience=5, min_delta=0)
validation_loss = 1000000000.0

for i in tqdm(range(75)):
    train_loss = 0
    val_loss = 0

    tqdm.write(f"Validation loss after {i} epochs is {validation_loss}")
    if early_stopper.early_stop(validation_loss):             
        break

    model.train()
    total_correct_train = 0
    total_samples_train = 0

    for images, labels, slice_number, patient_id, klg in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        slice_number = slice_number.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images, slice_number)

        labels = labels.view(-1, 1).float()

        # Compute the loss
        outputs = outputs.to(device)
        loss = (criterion(outputs, labels) * weights[labels.long()]).mean()  # correct loss function
        losses = loss.detach().cpu().numpy()
        train_loss += losses*len(labels)
        predictions = (outputs >= threshold).float()  # Convert to float for binary predictions

        loss.backward()
        optimizer.step()

        total_correct_train += (predictions == labels).sum().item()
        total_samples_train += labels.size(0)

    history["train"].append(train_loss / len(dataset_train))

    accuracy_test = total_correct_train / total_samples_train
    tqdm.write('train Accuracy: {:.2f}%'.format(accuracy_test * 100) + '\n')


    model.eval()
    val_loss = 0.0
    best_loss = 100000000
    with torch.no_grad():


        for val_images, val_labels, slice_number, patient_id, klg in val_loader:

            val_images = val_images.to(device)
            val_labels = val_labels.to(device)
            slice_number = slice_number.to(device)

            outputs = model(val_images, slice_number)

            val_labels = val_labels.view(-1, 1).float()

            loss = (criterion(outputs, val_labels) * weights[val_labels.long()]).mean()  # correct loss function
            losses = loss.detach().cpu().numpy()
            validation_loss = losses*len(val_labels)
            val_loss += losses*len(val_labels)

        history["validation"].append(val_loss / len(dataset_val))

        # early_stopping(val_loss, model)
        # if early_stopping.early_stop:
        #     print("Early Stopping")
        #     break


##################################################################################

model.eval()

# hashmap
# key: patient id
# value: array (160, 5) where each row is the logits associate to that slice
accumulate_logits = {}

# hashmap
# key: patient id
# value: true KLG
map_true_label = {}

all_targets = []
all_prediction = []

data_list = []

with torch.no_grad():

    total_correct_test = 0
    total_samples_test = 0

    for images, labels, slice_number, patient_id, klg in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        slice_number = slice_number.to(device)


        outputs = model(images, slice_number)
        
        labels = labels.view(-1, 1).float()

        # get the id of the patient
        id = patient_id[0]

        # assign the true "KLG" for the given id
        map_true_label[id] = int(labels[0].cpu())

        # To accumulate the predictions I have to handle two cases:
        # 1 first time I see a prediction for the id -> insert a new entry in the hasmap
        # 2 not the first time I see a prediction for the id -> update the matrix for the specific entry
        new_row = outputs.cpu()
        if id in accumulate_logits.keys():
            # If the ID exists, append the output list as a new row to the existing matrix
            matrix = accumulate_logits[id]
            new_row = outputs.cpu()
            updated_matrix = np.vstack((matrix, new_row[0]))
            accumulate_logits[id] = updated_matrix
        else:
            # If the ID doesn't exist, create a new matrix with the output list as the first row
            accumulate_logits[id] = new_row[0]

        loss = (criterion(outputs, labels) * weights[labels.long()]).mean()  # correct loss function

        predictions = (outputs >= threshold).float()

        # Iterate through the batch and append data to the list
        for i in range(len(patient_id)):
            data_list.append([patient_id[i], slice_number[i].cpu().item(), klg[i].item(), labels[i].cpu().item(), loss.item()])

        all_targets.append(labels.cpu()[0])
        all_prediction.append(predictions.cpu()[0])

        # Compute accuracy
        total_correct_test += (predictions == labels).sum().item()
        total_samples_test += labels.size(0)



accuracy_test = total_correct_test / total_samples_test
print('Test Accuracy: {:.2f}%'.format(accuracy_test * 100))

# Save the hashmap to a file
filename = "my_predictions" + str(seed) + ".pickle"
with open(filename, "wb") as file:
    pickle.dump((accumulate_logits, map_true_label), file)

plot_confMat(torch.stack(all_targets).numpy(), torch.stack(all_prediction).numpy())

plt.figure(figsize=(10,8)) # (width, height) is arbitrary size of your graph. e.g. (10,8)
plt.plot(range(1, len(history["train"]) + 1), history["train"], label="Training Loss")

# Plotting code for validation loss
plt.plot(range(1, len(history["validation"]) + 1), history["validation"], label="Validation Loss")

# Add labels, title, and legend
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()

# Save the graph as a PNG file
plt.savefig("./sen_bin_loss_graph_not_seg" + str(seed) +".png")

# cvs path
csv_file = 'test_results2.csv'

# Write the data to the CSV file
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    # Write the header row if needed
    writer.writerow(['Patient_ID', 'Slice_Number', 'KLG', 'Label', 'Loss'])
    # Write the data
    writer.writerows(data_list)

print(f'CSV file saved at {csv_file}')