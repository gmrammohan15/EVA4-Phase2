# Assignment

"Each" batch is supposed to collect 1000 images for the classes mentioned above: \
Google Drive Link had been emailed to you (one person in every group, you can add rest) already\
When you search, make sure to add some country names like "Small Quadcopter India". \
You cannot use these 3 names "USA", "India" and "China". We are trying to avoid downloading the same files. 
In all of these images, the objects MUST be flying and not on the ground (so no product images or them on the ground)\
Don't use Google Images only, use Flickr, Bing, Yahoo, DuckDuckGo\
Do not rename the files, as we'd like to know if the same files are there. \
You need to add 1000 images to the respective folders on Google Drive before Wednesday Noon. \
Train (transfer learning) MobileNet-V2 on a custom dataset of 31000 images:\
21k train, 10k test, remove duplicated (same names)\
with 4 classes Small 4Copter, Large 4Copter, Winged Drones and Flying Birds\
model is trained on 224x224, images you'll download are not 224x224. Think and implement the best strategy\
Upload the model on Lambda, and keep it ready for future use (use the same S3 bucket). 

## Solution
Google drive link for images(training and test images for each class)

# Google Drive Link for dataset:
https://drive.google.com/drive/folders/1HM6nFXTDMWzWOXx09WDc3vgWChaTVLs4?usp=sharing



    "Flying_Birds": 8164,
    "Large_QuadCopters": 4886,
    "Small_QuadCopters": 3612,
    "Winged_Drones": 5531


# Data preprocessing
Used pytorch transforms to resize the images to 224x224.
Sample Code, \
        transforms.Resize(300),\
        transforms.RandomCrop (224, pad_if_needed=True),\
        


# Model Training: 

Used standard mobilenet model from pytorch.
model = models.mobilenet_v2 (pretrained=True)
model.classifier[1] = torch.nn.Linear(in_features=model_ft.classifier[1].in_features, out_features=4)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

https://github.com/gmrammohan15/EVA4-Phase2/blob/master/mobilenet/EvaMobileNetDrones.ipynb


# Sample Images for training

![image](https://github.com/gmrammohan15/EVA4-Phase2/blob/master/mobilenet/images/AugmentedImageInput.png)


# Loss graph

![image](https://github.com/gmrammohan15/EVA4-Phase2/blob/master/mobilenet/images/TrainTestLoss.png)

# Accuracy graph

![image](https://github.com/gmrammohan15/EVA4-Phase2/blob/master/mobilenet/images/traintestAcc.png)


# Misclassified images

https://github.com/gmrammohan15/EVA4-Phase2/blob/master/mobilenet/EvaMobileNetDrones.ipynb


Winged Drones 

![image](https://github.com/gmrammohan15/EVA4-Phase2/blob/master/mobilenet/images/WingedDrones.png)


Flying Birds

![image](https://github.com/gmrammohan15/EVA4-Phase2/blob/master/mobilenet/images/FlyingBirds.png)


LargeQuad

![image](https://github.com/gmrammohan15/EVA4-Phase2/blob/master/mobilenet/images/LargeQuad.png)

SmallQuad

![image](https://github.com/gmrammohan15/EVA4-Phase2/blob/master/mobilenet/images/SmallQuad.png)
