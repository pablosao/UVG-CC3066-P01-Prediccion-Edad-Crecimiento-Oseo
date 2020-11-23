import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error
import numpy as np
import cv2
import pandas as pd
import glob
import random
import warnings

warnings.filterwarnings("ignore")
device = torch.device("cpu")

training_images = 'train/'
training_data = 'bone_data/boneage-training-dataset.csv'
testing_images = 'test/'
validation_images = 'validation/'

means = []
standard_deviations = []

image_names = glob.glob(training_images + '*.png')
random_images = random.sample(population=image_names, k=100)

for image_name in random_images:
    image = cv2.imread(image_name, 0)
    image = cv2.resize(image, (500, 500))
    mean, standard_deviation = cv2.meanStdDev(image)

    means.append(mean[0][0])
    standard_deviations.append(standard_deviation[0][0])

bone_age_df = pd.read_csv(training_data)
bone_age_df.iloc[:, 1:3] = bone_age_df.iloc[:, 1:3].astype(np.float)
training_df = bone_age_df.iloc[:len(image_names), :]
validation_df = bone_age_df.iloc[len(image_names):len(image_names) + 1500, :]
testing_df = bone_age_df.iloc[len(image_names) + 1500:, :]


class DatasetTransform(Dataset):
    def __init__(self, dataframe, image_dir, transform):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx):
        img_name = self.image_dir + str(self.dataframe.iloc[idx, 0]) + '.png'
        img = cv2.imread(img_name, 0)
        gender = np.atleast_1d(self.dataframe.iloc[idx, 2])
        bone_age = np.atleast_1d(self.dataframe.iloc[idx, 1])

        return self.transform({'image': img.astype(np.float64), 'gender': gender, 'bone_age': bone_age})


class DatasetToTensor(object):
    def __call__(self, sample):
        img, gender, bone_age = cv2.resize(sample['image'], (500, 500)), sample['gender'], sample['bone_age']

        return {'image': torch.from_numpy(np.expand_dims(img, axis=0)).float(),
                'gender': torch.from_numpy(gender).float(),
                'bone_age': torch.from_numpy(bone_age).float()}


class NormalizeDataset(object):
    def __init__(self, age_min, age_max):
        self.mean = mean
        self.standard_deviation = standard_deviation
        self.age_min = age_min
        self.age_max = age_max

    def __call__(self, sample):
        img, gender, bone_age = sample['image'], sample['gender'], sample['bone_age']
        img -= self.mean
        img /= self.standard_deviation
        bone_age = (bone_age - self.age_min) / (self.age_max - self.age_min)

        return {'image': img,
                'gender': gender,
                'bone_age': bone_age}


class Bottleneck(nn.Module):
    def __init__(self, in_nodes, nodes, sample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_nodes, nodes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(nodes)
        self.conv2 = nn.Conv2d(nodes, nodes, kernel_size=4, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(nodes)
        self.conv3 = nn.Conv2d(nodes, nodes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(nodes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.sample = sample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.sample is not None:
            residual = self.sample(x)

        out += residual
        return self.relu(out)


class AgePredictor(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.in_nodes = 64
        super(AgePredictor, self).__init__()
        self.conv1 = nn.Conv2d(1, self.in_nodes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_nodes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, self.in_nodes, layers[0])
        self.layer2 = self._make_layer(block, self.in_nodes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, self.in_nodes * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, self.in_nodes * 8, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.in_nodes * 8 * block.expansion, 400)
        self.res_relu = nn.ReLU()
        self.gen_fc_1 = nn.Linear(1, 16)
        self.gen_relu = nn.ReLU()
        self.cat_fc = nn.Linear(16 + 400, 200)
        self.cat_relu = nn.ReLU()
        self.final_fc = nn.Linear(200, num_classes)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, nodes, blocks, stride=1):
        sample = None
        if stride != 1 or self.in_nodes != nodes * block.expansion:
            sample = nn.Sequential(
                nn.Conv2d(self.in_nodes, nodes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(nodes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_nodes, nodes, stride, sample))
        self.in_nodes = nodes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_nodes, nodes))

        return nn.Sequential(*layers)

    def forward(self, x, y):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.res_relu(x)
        x = x.view(x.size(0), -1)

        y = self.gen_fc_1(y)
        y = self.gen_relu(y)
        y = y.view(y.size(0), -1)

        z = torch.cat((x, y), dim=1)
        z = self.cat_fc(z)
        z = self.cat_relu(z)
        z = self.final_fc(z)
        return self.sigmoid(z)


data_transform = transforms.Compose([
    NormalizeDataset(np.min(bone_age_df['boneage']), np.max(bone_age_df['boneage'])),
    DatasetToTensor()
])

training_dataset = DatasetTransform(dataframe=training_df, image_dir=training_images, transform=data_transform)
validation_dataset = DatasetTransform(dataframe=validation_df, image_dir=validation_images, transform=data_transform)
testing_dataset = DatasetTransform(dataframe=testing_df, image_dir=testing_images, transform=data_transform)

training_data_loader = DataLoader(training_dataset, batch_size=4, shuffle=False, num_workers=4)
validation_data_loader = DataLoader(validation_dataset, batch_size=4, shuffle=False, num_workers=4)
testing_data_loader = DataLoader(testing_dataset, batch_size=4, shuffle=False, num_workers=4)

age_predictor = AgePredictor(block=Bottleneck, layers=[4, 2, 20, 4], num_classes=1).to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(age_predictor.parameters(), lr=0.001, momentum=0.8)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)


@torch.no_grad()
def evaluate_model(model, data_loader):
    model.eval()

    for batch_no, batch in enumerate(data_loader):
        optimizer.zero_grad()

        img = batch['image'].to(device)
        gender = batch['gender'].to(device)
        outputs = model(img, gender)
        predictions = outputs.cpu().numpy()
        predictions = predictions.reshape(predictions.shape[0])

    return denormalize(predictions, np.min(bone_age_df['boneage']), np.max(bone_age_df['boneage']))


@torch.set_grad_enabled
def train_model(model, criterion, optimizer, scheduler, num_epochs=1):
    for epoch in range(num_epochs):
        scheduler.step()
        model.train()
        running_loss = 0.0
        val_running_loss = 0.0

        for batch_no, batch in enumerate(training_data_loader):
            img = batch['image'].to(device)
            gender = batch['gender'].to(device)
            age = batch['bone_age'].to(device)
            optimizer.zero_grad()

            outputs = model(img, gender)
            loss = criterion(outputs, age)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * img.size(0)

            if (batch_no + 1) % 25 == 0:
                print('Epoch {} - Batch {} '.format(epoch + 1, batch_no + 1))

        total_loss = running_loss / len(image_names)
        print('\n \n Epoch {} - Loss: {:.4f} \n \n'.format(epoch + 1, total_loss))

    model.eval()
    for val_batch in validation_data_loader:
        img = val_batch['image'].to(device)
        gender = val_batch['gender'].to(device)
        age = val_batch['bone_age'].to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(False):
            outputs = model(img, gender)
            loss = criterion(outputs, age)

        val_running_loss += loss.item() * img.size(0)

    print('Validation Loss: {:.4f}'.format(val_running_loss / 1500))

    return model


resnet_model = train_model(age_predictor, criterion, optimizer, scheduler, num_epochs=10)


def denormalize(inputs, age_min, age_max):
    return inputs * (age_max - age_min) + age_min


result_array = evaluate_model(age_predictor, testing_data_loader)
testing_df['output'] = result_array
testing_df['output'] = np.round(testing_df['output'], decimals=2)
test_df = testing_df.reset_index()
rmse = np.sqrt(mean_squared_error(test_df['boneage'], test_df['output']))


def display_predictions(num):
    idx = random.sample(range(0, test_df.shape[0]), num)
    for i in idx:
        img = cv2.imread(testing_images + str(test_df['id'][i]) + '.png')
        img = cv2.resize(img, (500, 500))
        cv2.putText(img, 'Actual:' + str(test_df['boneage'][i]), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
        cv2.putText(img, 'Predicted:' + str(test_df['output'][i]), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255))
        cv2.imwrite(num + '.png', img)
        cv2.imshow('Bone Age Prediction', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


display_predictions(25)
