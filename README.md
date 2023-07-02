# Deep_fake_face_detection_CNN_neural_network


DeepFake is composed from Deep Learning and Fake and means taking one person from an image or video and replacing with someone else likeness using technology such as Deep Artificial Neural Networks. Large companies like Google invest very much in fighting the DeepFake, this including release of large datasets to help training models to counter this threat.The phenomen invades rapidly the film industry and threatens to compromise news agencies. Large digital companies, including content providers and social platforms are in the frontrun of fighting Deep Fakes. GANs that generate DeepFakes becomes better every day and, of course, if you include in a new GAN model all the information we collected until now how to combat various existent models, we create a model that cannot be beatten by the existing ones.

First we will work on detecting faces that were forged and we will work on developing a model to detect videos.


Image Dataset
This dataset contains faces extracted from deepfake-detection-challenge. All images were of size 224x224.

Due to memory issue we will only use a sample of the entire dataset for prediction.


About Video the dataset
Files

train_sample_videos.zip - a ZIP file containing a sample set of training videos and a metadata.json with labels. the full set of training videos is available through the links provided above.
sample_submission.csv - a sample submission file in the correct format.
test_videos.zip - a zip file containing a small set of videos to be used as a public validation set. To understand the datasets available for this competition, review the Getting Started information.
Metadata Columns

filename - the filename of the video
label - whether the video is REAL or FAKE
