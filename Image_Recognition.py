from imageai.Prediction import ImagePrediction #import Image AI libray
import os


path = os.getcwd()
prediction = ImagePrediction()
prediction.setModelTypeAsInceptionV3() # Set model type as Inception (becuase I use the Inception model)
prediction.setModelPath(os.path.join(path, "inception_v3_weights_tf_dim_ordering_tf_kernels.h5")) # Put the Inception model's path
prediction.loadModel()

predictions, probabilities = prediction.predictImage(os.path.join(path, "file.jpg"), result_count=5 ) # Here input the file name
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(f"According to the program, there is a {eachProbability}% chance that this is an {eachPrediction}") # The program prints out the results
