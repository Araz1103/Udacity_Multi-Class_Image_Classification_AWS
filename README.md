# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. We have used the Dogs Breed Classification Dataset for the same

## Project Set Up and Installation
1. Enter AWS through the gateway in the course and open SageMaker Studio. 
2. Download the starter files.
3. Download/Make the dataset available in your S3.
4. Open the `train_and_deploy.ipynb` and follow the steps there. You can verify if data is uploaded correctly also with this. 
5. The supporting files `hpo_gpu.py` and `train_model.py` & `inference.py` need to be in the same directory as train_and_deploy notebook

## Dataset
The provided dataset is the dogbreed classification dataset which can be found in the link in `train_and_deploy.ipynb`
This project is designed to be dataset independent so if there is a dataset that is more interesting or relevant to your work, you are welcome to use it to complete the project.

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 

## Hyperparameter Tuning

We are using a Resnet 50 Pre-Trained Model with added fully connected layers, which we are wanting to fine tune on our dataset. To do that, we want to find some optimal hyper paramters to train with, so we use HP Tuner from SageMaker. 

For tuning Hyper Parameters, I decided to tune the "Learning Rate" & the "Batch Size" Parameters. The reason to do this is that the Learning Rate is often the one responsible for how well the model trains, since an extremely small one can delay training, and a very large one can result in not finding the optima. It is also a very tricky parameter to tune manually, and hence a HP search can help us. We are trying LR from 0.001 till 0.1

Similiarly, we tune Batch size as an optimal Batch Size can help the loss decrease faster & learn the weights better. I have provided a good range of Batch Sizes, from 16 till 256, since different models do well with different Batch Sizes

One can also tune other HPs like number of Epochs, number of layers in the Model or anything which is difficult to decide manually

The Best HPs we got for this were:

- Batch Size: 16
- Learning Rate: 0.0041

## Screen Shots

All screenshots of the completed tranining jobs, log metrics & deployed endpoint are present in the `screenshots` folder in the repository

Remember that your README should:
- Include a screenshot of completed training jobs
- Logs metrics during the training process
- Tune at least two hyperparameters
- Retrieve the best best hyperparameters from all your training jobs

## Debugging and Profiling
For this, we use the SageMaker Debugger and Profiler, to give us more insights about our training process. We have added the hooks in `train_model.py` to use these. In the `train_and_deploy.ipynb` we define the rules for the debugger and profiler, and also to generate the Profiler Report which is present in the `ProfilerReport` folder in the Repo. 

**TODO**: Give an overview of how you performed model debugging and profiling in Sagemaker

### Results

Some Valuable Insights from Debugger and Profiler were:
- We could try early stopping, since the training loss is increasing after 1000 steps, as seen in the debugger plot. 
- We could try to train for more epochs to get more data, and decide if loss is decreasing in the further epochs
- We see from the Profiler reports that we are under utilising the GPU
- We also see from the Profiler Reports that we can try higher batch sizes

**TODO**: What are the results/insights did you get by profiling/debugging your model?

**TODO** Remember to provide the profiler html/pdf file in your submission.


## Model Deployment

See the Deployment section in `train_and_deploy.ipynb` notebook for the deployment code. 

- We use the `inference.py` script, to help generate a Pytorch model for Image Prediction, from the Model we have gotten with training with the best Hyper Parameters.
- We create an endpoint with this Pytorch model on a single ml.t2.medium instance. 
- Then we fetch a sample input for querying from a hosted S3 Bucket.
- We use predictor.predict() method to get the most likely dog class from the model
- We can then check if the class predicted is accurate for the given image input

**TODO**: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

**TODO** Remember to provide a screenshot of the deployed active endpoint in Sagemaker.
