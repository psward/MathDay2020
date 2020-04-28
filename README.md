# MathDay2020

Above are my two, three minute talks for Math Day 2020! 

YPou can view <i>Big Jumps or Little Steps: Fighting Gerrymandering with Random Walks</i> (here)[https://youtu.be/SwfkuVxsSSs].

You can view <i>Imposter Syndrome: Classifying Salamanders with Computer Vision and Artificial Neural Networks</i> [here](https://youtu.be/m2p13IJojD8).

You can view <i>Using Unsupervised Documenting Clustering to Facilitate Coronavirus Research</i> (https://youtu.be/RePL5Ebn73A)[here].



## Big jumps or little steps: fighting gerrymandering with random walks

Co-Presentor: Suzie Tovar

Political gerrymandering is a complex and pressing threat to our system of government. At the heart of our difficulties to fairly divide ourselves in voting district lies a math problem – how do we measure fairness? How can we use that measure to help draw fair district boundaries? Our project is part of nationwide collaboration of mathematicians, demographers, lawyers, mapmakers, political leaders, and citizens attempting to develop tools for this purpose. We will survey Markov Chain Monte Carlo (MCMC) methods used successfully in the PA Supreme Court case, work to make MCMC more widely available via the Python package GerryChain, and a recent improvement to MCMC called recombination.

## Imposter Syndrome: Classifying Salamanders with Computer Vision and Artificial Neural Networks

Salamanders serve as important tetrapod models for developmental, regeneration and evolutionary studies. In order to tell certain species of salamander apart, genetic sequencing is used which can be difficult as the sequence could be rather lengthy and takes time and equipment many places do not have. There are also many museums with specimens for which DNA sequencing cannot be done, thus the need for another method for identifying salamanders. This project aims to create a deep learning model that is able to take in a picture of a salamander and return its classification.

To do this we built a Convolution Neural Network and supplied it with high resolution images provided by Washington University and the Smithsonian. With these images and five classifications of salamanders, the model is able to predict with high accuracy the correct classification of salamander. With more images and higher-performance computers, the model could produce more accurate predictions.


## Using Unsupervised Documenting Clustering to Facilitate Coronavirus Research

With COVID19 constantly looming over us and lots of questions with very little answers, the White House and a coalition of leading research groups have prepared the COVID-19 Open Research Dataset (CORD-19). CORD-19 is a resource of over 52,000 scholarly articles, including over 41,000 with full text, about COVID-19, SARS-CoV-2, and related coronaviruses. With this they released a call to action with ten open questions, all related to how COVID 19 operates. Before we can begin answering these questions, it would be nice if we could find some way to cluster these documents into groups of related topics so that when we have a question related to immunizations, we don't also have to parse through documents about transmission rates in animals. Using Topic Modeling techniques with stochastic block modeling, I show that you can produce a hierarchy of document clustering, where you will also receive the key words that most describe those clusters. This mechanism will provide a fast and efficient way to parse the information, and only get the articles related to what is important for your study. 


### Supplementary Information
For each talk you can go here: https://mathday2020.imfast.io/ to open and view the notebooks used for each project.  I can provide the actual notebooks upon request for anyone interested!  They are currently locked up in a docker image I can't deploy yet because I have other code running and the instance needs 18 gigs of RAM to launch, or I will lose the hSBM model.

To view the notebooks, click on the html file, click on raw, and then save this as an html file.  Once saved, open in your favorite browser!
