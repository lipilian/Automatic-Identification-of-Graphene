# Automatic-Identification-of-Graphene
Graphene synthesis is a very promising field. Graphene is the strongest material ever tested, one of the highest thermal conductivities, and is nearly transparent. Graphene fabrication is very involved, so there is high demand for finding a way to identify graphene in electron microscope images. 
Our objective is to detect graphene surface coverage in SEM (scanning electron microscope) images to quantify the efficacy of different synthesis recipes. We will use the tool provided by the nanoMFG node to generate training data by manually determine the areas covered by graphene. Eventually, we will develop a machine learning algorithm to automatically scan SEM images to determine the locations and features of graphene. 
Our solution is to use the provided package software to manually mask the graphene area then read the binary image to be processed as our test labels. We will also read the raw image to be processed for feature generation. 

![raw](https://github.com/lipilian/Automatic-Identification-of-Graphene/blob/master/raw.png)

For feature generation, we use an 8x8 sliding window to slide by 4 pixels at a time to generate numerous 8x8 patches for the raw image as matrices. We then categorize the color intensities represented by numbers in the matrices into 127 bins as our 127 features. Each 8x8 patch makes up one layer of training data so one .TIF image provides us with large amounts of training data. 

![hist](https://github.com/lipilian/Automatic-Identification-of-Graphene/blob/master/hist.png)

The histogram in Figure 2 shows two peaks, one corresponding to the graphene sections of the image and the other corresponds to the rest. Most of the histograms are able to seperate the peaks fairly easily. For every two steps in intensity we fit the data into one feature, so for all 0-255 intensity levels we obtain 127 features. 

For label generation, we use the same sliding window algorithm to generate an 8x8 binary matrix for each patch. We then perform max-pooling for each patch, obtaining a zero or one label for each patch, 0 representing graphene, 1 representing background. 


![QDA](https://github.com/lipilian/Automatic-Identification-of-Graphene/blob/master/QDA.png)

We used five .TIF raw images and their masked images for the training. We performed standardization on the data to compensate for lighting differences between images. We then use PCA to select two best features. We used Gaussian kernel and QDA classifier and found QDA classifier returning the better result, visualized in Figure 3. 

![result](https://github.com/lipilian/Automatic-Identification-of-Graphene/blob/master/result.png)
