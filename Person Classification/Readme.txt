This Project is developed by Gaurav Verma.

Please read it thoroughly to understand its working.

- How to use:

1. Fist step is to collect data in the form of images and put it in separate forlders.
   I have already collected some. You can find them in the folders.
 
2. After collecting data, use DataCleaning_Saving_Cropped_face_images.py to obtain the data we need in a cleaner format.
   use savCroppedImages() function to put data in a separate folder. I have already done this in TC_crop, KS_crop and VK_crop. Take a look.
 
3. Use - ImageProcessing_Model_Training.py. 
   Now we have a clean form of data, that is faces of those persons only. Now we need to process those images to train out model.
   After processing them, I have tested 3 models - Logistic Regression, SVM, Random Forest. I selected SVM based on its accuracy and commented the other two.
   You can try them out. Then I have save the model in Person_Classification_SVM_MODEL_JOblib.pkl file.

4. Now it is time to test our data. 
   Use - Testing_Model.py
   Now load the Person_Classification_SVM_MODEL_JOblib.pkl file using joblib
   Initialize the ImagePath variable and run the code. It will show the name of the person in the output window.

5. That's it. You can further modify it according to your requirements. Or upload it on a Flask server.


- Error you might encounter:

1. Memory error - 
   You will get his error while writing images. Try no to write all images at once. Do it in parts.

2. StandardScaler error -
   This is not a compilation or Interpretation error. 
   Using StandardScaler, it completely changed my results. You can use it or comment it according to your need.


Thank You for Reading :)  
	
 
	
