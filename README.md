# privacy_analysis
We used a simple metric-based membership inference attack (MIA) in an unsupervised manner. 

We treated the MIA as a classification task: the target is to classify whether a specific real image is 0 (not presented in the training dataset) or 1 (presented in the training dataset).

We presume that, the synthetic records must bear similarity with the records that were used to generate them [22]. For a real image, we calculated the similarities of all synthetic images between this image. If the mean similarity between this real image and all synthetic images is under a specific threshold, then this real image is considered as 1 (presented in the training dataset). We performed this analysis on all real images that were used to train the synthetic image (whose ground truth labels are all 1) and calculated the error of our metric-based MIA. A privacy preserved model should have high error. We define this error as MIA safe score. 
