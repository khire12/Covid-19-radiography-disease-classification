import torch
import torchvision
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt




train_dataset = ChestXRayDataset(train_dirs, train_transforms)
test_dataset = ChestXRayDataset(test_dirs,test_transforms)

train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=6, shuffle=True)
test_dl = torch.utils.data.DataLoader(test_dataset,batch_size = 6, shuffle = True)

def show_preds():
  resnet18.eval()
  images, labels = next(iter(test_dl))
  images = images.to(device)
  labels = labels.to(device)
  outputs = resnet18(images)
  print(outputs.shape)
  _, preds = torch.max(outputs, 1)
  show_images(images.cpu(), labels.cpu(), preds.cpu())



class_names = train_dataset.class_names
def show_images(images, labels, preds):
  plt.figure(figsize= (8,4))
  for i, image in enumerate(images):
    plt.subplot(1, 6, i+1, xticks=[], yticks=[])
    image = image.numpy().transpose((1,2,0))
    mean = np.array([0.485,0.456, 0.406])
    std = np.array([0.229, 0.224,0.225])
    image = image*std + mean
    image= np.clip(image, 0., 1.)
    plt.imshow(image)
    col ='green'
    if preds[i]!=labels[i]:
      col = 'red'
    
    plt.xlabel(f'{class_names[int(labels[i].numpy())]}')
    plt.ylabel(f'{class_names[int(preds[i].numpy())]}', color = col)
  plt.tight_layout()
  plt.show()
  

images, labels = next(iter(train_dl))
show_images(images,labels, labels)




resnet18 = torchvision.models.resnet18(pretrained=True)
resnet18.fc = torch.nn.Linear(in_features = 512, out_features=3)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet18.parameters(), lr = 3e-5)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    resnet18.cuda()
resnet18 = resnet18.to(device)

def train(epochs):
  print(f'Start training...')
  for e in range(0, epochs):
    print('='*20)
    print(f'Start epoch {e+1}/{epochs}')
    print('='*20)
    train_loss = 0
    resnet18.train()
    for train_step, (images, labels) in enumerate(train_dl):
      images = images.to(device)
      labels = labels.to(device)
      optimizer.zero_grad()
      outputs = resnet18(images)
      loss = loss_fn(outputs, labels)
      loss.backward()
      optimizer.step()
      train_loss += loss.item()
      if train_step % 20 == 0:
        print('Evaluating at step', train_step)
        resnet18.eval()
        acc = 0.
        val_loss = 0.
        for val_step, (images, labels) in enumerate(test_dl):
          images = images.to(device)
          labels = labels.to(device)
          outputs = resnet18(images)
          loss = loss_fn(outputs, labels)
          val_loss += loss.item()

          _, preds = torch.max(outputs, 1)
          acc += sum((preds == labels))
        
        val_loss = val_loss/(val_step+1)
        acc = acc / len(test_dataset)
        print(f'Val loss:{val_loss:.4f}, Accuracy:{acc:.4f}')
        show_preds()
        resnet18.train()

        if acc > 0.95:
          print('Perfomance level satisfied...')
          return

    train_loss = train_loss/len(train_step+1)
    print(f'Training loss: {train_loss:.4f}')      