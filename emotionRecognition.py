# Imports
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from customDataset import EmotionDataset

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Convolutional Neural Network
class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1)) #keeps size
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride =(2,2)) #cuts size in half
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1)) #keeps size
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=(1,1)) #keeps size
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1),padding=(1, 1))  # keeps size
        self.fc1 = nn.Linear(64*12*12, num_classes)

    def forward(self, x):
        x = F.relu((self.conv1(x)))
        x = F.relu((self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x

# Saving model to file
def save_model(state, filename = "saved_model.pth.tar"):
    print("Saving model")
    torch.save(state, filename)

# Loading model to file
def load_model(file):
    print("Loading model")
    model.load_state_dict(file['state_dict'])
    optimizer.load_state_dict(file['optimizer'])

# Hyperparameters
in_channel = 1
num_classes = 6
learning_rate = 0.001
batch_size = 64
num_epochs = 55

# Load Data
train_set = EmotionDataset(root_dir = "Training", transform = transforms.ToTensor(), train=True)
test_set = EmotionDataset(root_dir = "Testing", transform = transforms.ToTensor(), train=False)
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=0)

# init network
model = ConvolutionalNeuralNetwork(in_channels=in_channel, num_classes=num_classes).to(device)

# loss - cost function
criterion = nn.CrossEntropyLoss()

# learning algorithm
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

# training model
def train_model():
    print("Training Model")
    for epoch in range(num_epochs):
        losses = []
        for batch_idx, (data, targets) in enumerate(train_loader):
            # get data in cpu
            data = data.to(device=device)
            targets = targets.to(device=device)

            # forward
            scores = model(data)
            loss = criterion(scores, targets)

            losses.append(loss.item())

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()

        print(f"Cost at epoch {epoch} is {sum(losses) / len(losses)}")

# checks accuracy after training on training and testing data
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        if loader.dataset.train:
            print("Checking accuracy on training data")
        else:
            print("Checking accuracy on test data")
        for x, y in loader:
            x = x.to(device = device)
            y = y.to(device=device)
            #x = x.reshape(x.shape[0], -1) #for NN

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')

    model.train()

if __name__ == "__main__":
    # training model
    # train_model()

    # model_save = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
    # save_model(model_save)

    # check accuracy
    # check_accuracy(train_loader, model)
    # check_accuracy(test_loader, model)

    # load and use model
    load_model(torch.load('saved_model.pth.tar'))

    # processes each frame for model
    def image_loader(image):
        # transformations
        r = torchvision.transforms.Resize((48, 48))
        t = transforms.ToTensor()

        # processing
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = t(image)
        image = r(image)
        image = Variable(image, requires_grad=True)
        image = image.unsqueeze(0)
        return image.cpu()

    # processes each frame for display
    def image_processor(im, text):
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        im = cv2.resize(im, (729, 729))
        cv2.putText(im, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (9, 9, 186), 2, cv2.LINE_AA)
        return im

    # determines confidence percent
    def soft_max(tens):
        sm = torch.nn.functional.softmax(tens, dim=0)
        max_sm = torch.max(sm * 100).item()
        return int(round(max_sm, 0))


    emotions_dict = {0: "Anger", 1: "Fear", 2: "Happiness", 3: "Neutral", 4: "Sadness", 5: "Surprise"}
    video = cv2.VideoCapture(0)
    model.eval()

    with torch.no_grad():
        while True:
            result, vid = video.read()

            if result is False: break

            # processing
            image = image_loader(vid)
            text = (str(emotions_dict[torch.argmax(model(image)).item()]) + " " + str(soft_max(model(image).squeeze(0))) + "%")

            # display
            cv2.imshow("Emotion Detection", image_processor(vid, text))

            k = cv2.waitKey(1) & 0xFF

            # esc to end
            if k == 27:
                break
