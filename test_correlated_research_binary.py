import os
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
import torchvision.transforms as transforms
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm
import sklearn.metrics as metrics
from torch.utils.data import Dataset, DataLoader
#from pytorchtools import EarlyStopping
import torch.optim as optim
from networks.vit_seg_modeling import Model3D_CNN

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
        
def plot_confMat(target, predicted, seed, out_parentDir="./confusion_matrix_result"):
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
    plt.savefig(f"{out_parentDir}/correlated_research_binary" + str(seed)+ ".png", bbox_inches="tight", pad_inches=0)
    print(f"Confusion Matrix saved in: {out_parentDir}/correlated_research_binary" + str(seed)+ ".png")

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


class CustomDataset(Dataset):
    def __init__(self, ids_list, transform=None, is_training = False):

        self.transform = transform
        self.ids_list = ids_list

        image_sum = 0
        image_squared_sum = 0
        count = 0
        # I assume the image shape is (2, 384, 384), and label shape is (1,). 
        # During training I compute the image mean and standard deviation for both channels
        # if (is_training):
        #     for file in self.npz_files:
        #         data = np.load(self.directory + file)
        #         image = data['image_raw']
                
        #         # Convert images to tensors. 
        #         # TODO how can I convert the first array without loosing information since torch does not support
        #         # uint16 as data type
        #         # image_raw = torch.tensor(image_raw, dtype=torch.float32)
        #         # image_seg = torch.tensor(image_seg, dtype=torch.uint8)         

        #         # Create 2-channel array
        #         image_sum += np.mean(image)
        #         image_squared_sum += np.mean(image * image)
        #         count += 1
        #     # shape: (2, 1, 1)
        #     self.image_mean = image_sum / count 
        #     image_squared_mean = image_squared_sum / count
            
        #     # shape: (2, 1, 1)
        #     image_variance = image_squared_mean - (self.image_mean * self.image_mean) 
        #     self.image_std = np.sqrt(image_variance)

        #     # Save these values globally to reuse them to standardize validation and test set
        #     global global_image_mean
        #     global_image_mean = self.image_mean
        #     global global_image_std
        #     global_image_std = self.image_std
        # else:
        #     self.image_mean = global_image_mean
        #     self.image_std = global_image_std


    def __len__(self):
        return len(self.ids_list)

    def __getitem__(self, idx):
        id = self.ids_list[idx]
        image = id_slices_tensors[id]
        klg = id_klg_dict[id]
        label = 0
        if(klg > 1):
            label = 1
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
        
        # apply the transformation if it is defined
        if self.transform:
            return_image = self.transform(image)
        # Standardization
        # return_image = (return_image - self.image_mean) / self.image_std

        # converts the image tensor's data type to float
        return_image = return_image.float()

        # return the output
        return return_image, label


data_folder = '/home/manfre/workspace/TransUnet/data/Synapse/true_selected_slices/'
folder_path = '/home/manfre/workspace/TransUnet/data/Synapse/true_selected_slices/*.npz'

seed = 141312
random.seed(seed)

# define the global variable for mean and standard deviation.
# these variable will be used for standardization procedure
global_image_mean = 0.0
global_image_std = 0.0
# define the output size
output_size = (160, 160)

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

training_ids = positive_ids[:positive_ids_per_part*num_parts - positive_ids_per_part*7] + negative_ids[:negative_ids_per_part*num_parts - negative_ids_per_part*7]
validation_ids = positive_ids[positive_ids_per_part*num_parts - positive_ids_per_part*7: positive_ids_per_part*num_parts - positive_ids_per_part*3] + negative_ids[negative_ids_per_part*num_parts - negative_ids_per_part*7: negative_ids_per_part*num_parts - negative_ids_per_part*3]
testing_ids = positive_ids[positive_ids_per_part*num_parts - positive_ids_per_part*3: ] + negative_ids[negative_ids_per_part*num_parts - negative_ids_per_part*3: ]

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



my_transform = transforms.Compose([
                                transforms.Lambda(custom_transform),
                                transforms.CenterCrop(output_size), # 160x160
                                ])


dataset_train = CustomDataset(ids_list=training_ids, transform=my_transform)
dataset_val = CustomDataset(ids_list=validation_ids, transform=my_transform)
dataset_test = CustomDataset(ids_list=testing_ids, transform=my_transform)

# create loaders
train_loader = DataLoader(dataset_train, batch_size=15, shuffle=True)
val_loader = DataLoader(dataset_val, batch_size=15, shuffle=False)
test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False)


model = Model3D_CNN(120, residual_channel=128, is_binary_problem=True)
print(sum(p.numel() for p in model.parameters() if p.requires_grad))
print(seed)
# starting 32 finish with 128... with  8181893 parameters -> 69.41% - 70.59%
# starting 32 finish with 128... with  4862501 parameters -> 72.58% - 70.97% - 69.35%
# starting 64 finish with 256... with 19139653 parameters -> 53.23%
# starting 16 finish with 64... with   1256725 parameters -> 54.84% - 56.45% - 67.74% - 70.59% - 74.19%

# Define the BCEWithLogitsLoss function with weight parameter
# weights = torch.Tensor([0.61, 1 - 0.61])  # weight for [0 (negative), 1(positive)]
# criterion = nn.BCELoss(reduction="none")
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#weights = weights.to(device)

threshold = 0.5
model = model.to(device)
#early_stopping = EarlyStopping(patience=5)
history = {"train":[], "validation":[]}

all_targets = []
all_prediction = []

early_stopper = EarlyStopper(patience=10, min_delta=0.2)
validation_loss = 1000000000.0
for i in tqdm(range(75)):
    train_loss = 0

    if early_stopper.early_stop(validation_loss):             
        break
    print(validation_loss)
    model.train()
    total_correct_train = 0
    total_samples_train = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        

        # Forward pass
        outputs = model(images)
        labels = labels.view(-1, 1).float()

        # Compute the loss
        # loss = (criterion(outputs, labels) * weights[labels.long()]).mean()
        loss = criterion(outputs, labels)
        losses =  loss.detach().cpu().numpy()
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
    
    best_loss = 100000000
    val_loss = 0.0
    with torch.no_grad():


        for val_images, val_labels in val_loader:

            val_images = val_images.to(device)
            val_labels = val_labels.to(device)

            outputs = model(val_images)

            val_labels = val_labels.view(-1, 1).float()

            # loss = (criterion(outputs, val_labels) * weights[val_labels.long()]).mean()
            loss = criterion(outputs, val_labels)
            losses = loss.detach().cpu().numpy()
            validation_loss = losses*len(val_labels)
            val_loss += losses*len(val_labels)

        history["validation"].append(val_loss / len(dataset_val))




##################################################################################

model.eval()

with torch.no_grad():

    total_correct_test = 0
    total_samples_test = 0

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)


        outputs = model(images)

        labels = labels.view(-1, 1).float()

        loss = criterion(outputs, labels)

        predictions = (outputs >= threshold).float()  # Convert to float for binary predictions

        all_targets.append(labels.cpu()[0])
        all_prediction.append(predictions.cpu()[0])

        # Compute accuracy
        total_correct_test += (predictions == labels).sum().item()
        total_samples_test += labels.size(0)



accuracy_test = total_correct_test / total_samples_test
print('Test Accuracy: {:.2f}%'.format(accuracy_test * 100))

plot_confMat(torch.stack(all_targets).numpy(), torch.stack(all_prediction).numpy(), seed)

print(len(history["train"][9:]))
print(len(history["validation"][9:]))
plt.figure(figsize=(10,8)) # (width, height) is arbitrary size of your graph. e.g. (10,8)
plt.plot(range(10, len(history["train"]) + 1), history["train"][9:], label="Training Loss")

# Plotting code for validation loss
plt.plot(range(10, len(history["validation"]) + 1), history["validation"][9:], label="Validation Loss")

# Add labels, title, and legend
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()

# Save the graph as a PNG file
plt.savefig("./correlated_research_bin_loss_graph.png")


print(f"These are the F1 Macro value: {f1_score(torch.stack(all_targets).numpy(), torch.stack(all_prediction).numpy(), average='macro')*100 :.2f}%")