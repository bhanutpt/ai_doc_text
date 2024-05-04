# ai_doc_text
An AI/ML model generation for detecting hand written text.  

The project is an experiment in training character recognitions from an image of hand written text.

The steps are acheived with simple programs which does a task at a time and so gives maximum control of the flow.

Here are the tasks and related documentation on how to utilize the programs.

1. Preprocess images for character extraction

Usage :-
python preprocess_images <source_folder> <destnation_folder>

Example :-
python preprocess_images.py src_images out_images 

Make sure to keep the input documents in a folder and specify input folder and output floder as command line arguments.
The input images should be a picture of document with handwritten characters.

https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
The above link has good example demonstrations which can be very useful for preprocessing.

It is observed that instead of extracting characters and then applying preprocessing,
Preprocessing can be performed to the full document image and then extract characters.

2. Generate Character dataset from a given image

Usage :-
python generate_chars_images.py <input_image_filename> <destnation_folder>

Example :-
python preprocess_images.py page_12.jpg page12_chars 

The program automatically generates a list of characters from a given image and saves them to the destination folder.

3. Generate standard size character images (32 x 32)

Usage :-
python get_std_size_chars_images.py <source_folder_name> <destination_folder_name>

Example :
python get_std_size_chars_images.py page12_chars page12_chars_dataset

Program to take input character images and standardize them to a given standard size images. It will be helpful to load them and process them in AI/ML model training.

4. Label the data and prepare dataset 

The character dataset prepared in the previous step should be labelled. The labelling is acheived with renaming the files with the characters in it.
Say we have '9' in character_72.png, then we have to rename it as 9-character_72.png, and '5' in charater_63.png, then rename it as 5-character_63.png and so on.

Once the labelling is complete, then we can combine labelled character images to a folder which can be used as input to model training.

5. Training

Usage :-
python train_model.py <source_folder_name> 

Example :
python train_model.py p1_chars_dataset

The program trains the model, and the model will be saved as 'model_output.h5'.

6. Testing

Usage :-
python test_trained_model.py <input_image_filename> 

Example :-
python test_trained_model.py character90.jpg 

The program will load model named 'model_output.h5' (generated in previous step) and run predictions on a given input image.
The program prints the prediction probabilities for each class and also prints predicted class.






