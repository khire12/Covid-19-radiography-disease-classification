import os
import torch
import torchvision
import numpy as np
from PIL import Image
import random
from matplotlib import pyplot as plt
torch.manual_seed(0)
print('Using Pytorch version ', torch.__version__)
from torch.utils.data import Dataset

class ChestXRayDataset(Dataset):
  def __init__(self, image_dirs, transform):
    def get_images(class_name):
      images = [x for x in os.listdir(image_dirs[class_name]) if x.endswith('png')]
      print(f'Found {len(images)} {class_name} examples')
      return images

    self.images = {}
    self.class_names =  ['normal', 'viral', 'covid']
    for c in self.class_names:
      self.images[c] = get_images(c)

    self.transforms = transform
    self.image_dirs = image_dirs

  def __len__(self):
    return sum([len(self.images[c]) for c in self.class_names])

  def __getitem__(self, index):
    class_name = random.choice(self.class_names)
    index = index % len(self.images[class_name])  
    image_name = self.images[class_name][index]
    image_path = os.path.join(self.image_dirs[class_name], image_name)
    image = Image.open(image_path).convert('RGB')
    return self.transforms(image), self.class_names.index(class_name)



train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size = (224, 224)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean = [0.485,0.456,0.406], std=[0.229, 0.224,0.225])
])

test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size = (224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean = [0.485,0.456,0.406], std=[0.229, 0.224,0.225])
])


train_dirs ={
    'normal':'/content/drive/My Drive/COVID-19 Radiography Database/normal',
    'viral':'/content/drive/My Drive/COVID-19 Radiography Database/viral',
    'covid':'/content/drive/My Drive/COVID-19 Radiography Database/covid'
}

test_dirs = {
    'normal':'/content/drive/My Drive/COVID-19 Radiography Database/test/normal',
    'viral':'/content/drive/My Drive/COVID-19 Radiography Database/test/viral',
    'covid':'/content/drive/My Drive/COVID-19 Radiography Database/test/covid'
}


train_dataset = ChestXRayDataset(train_dirs, train_transforms)
test_dataset = ChestXRayDataset(test_dirs,test_transforms)

train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=6, shuffle=True)
test_dl = torch.utils.data.DataLoader(test_dataset,batch_size = 6, shuffle = True)


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