# Automatic-Identification-of-Graphene
Graphene synthesis is a very promising field. Graphene is the strongest material ever tested, one of the highest thermal conductivities, and is nearly transparent. Graphene fabrication is very involved, so there is high demand for finding a way to identify graphene in electron microscope images. 
Our objective is to detect graphene surface coverage in SEM (scanning electron microscope) images to quantify the efficacy of different synthesis recipes. We will use the tool provided by the nanoMFG node to generate training data by manually determine the areas covered by graphene. Eventually, we will develop a machine learning algorithm to automatically scan SEM images to determine the locations and features of graphene. 
Our solution is to use the provided package software to manually mask the graphene area then read the binary image to be processed as our test labels. We will also read the raw image to be processed for feature generation. 

![raw](https://github.com/lipilian/Automatic-Identification-of-Graphene/blob/master/raw.png)
