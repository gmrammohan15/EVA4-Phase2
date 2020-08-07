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
Upload the model on Lambda, and keep it ready for future use (use the same S3 bucket). \

## Solution
Google drive link for images(training and test images for each class)

# Google Drive Link for dataset:
https://drive.google.com/drive/folders/1HM6nFXTDMWzWOXx09WDc3vgWChaTVLs4?usp=sharing



    "Flying_Birds": 8164,
    "Large_QuadCopters": 4886,
    "Small_QuadCopters": 3612,
    "Winged_Drones": 5531


# links:
Training: https://github.com/gmrammohan15/EVA4-Phase2/blob/master/mobilenet/EvaMobileNetDrones.ipynb
Misclassified image : https://github.com/gmrammohan15/EVA4-Phase2/blob/master/mobilenet/EvaMobileNetDrones.ipynb


# Sample Images for training

![image](https://github.com/gmrammohan15/EVA4-Phase2/blob/master/mobilenet/images/AugmentedImageInput.png)


# Loss graph

![image](https://github.com/gmrammohan15/EVA4-Phase2/blob/master/mobilenet/images/TrainTestLoss.png)

# Accuracy graph

![image](https://github.com/gmrammohan15/EVA4-Phase2/blob/master/mobilenet/images/traintestAcc.png)


# Misclassified images

![image](https://github.com/gmrammohan15/EVA4-Phase2/blob/master/mobilenet/images/WingedDrones.png)

![image](https://github.com/gmrammohan15/EVA4-Phase2/blob/master/mobilenet/images/FlyingBirds.png)

![image](https://github.com/gmrammohan15/EVA4-Phase2/blob/master/mobilenet/images/LargeQuad.png)

![image](https://github.com/gmrammohan15/EVA4-Phase2/blob/master/mobilenet/images/SmallQuad.png)
