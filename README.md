#Repository to test out creating a custom model for TensorFlow
###Training and Testing Instructions:

####Step 1: Convert the input dataset to a common resolution

        python rename.py
        
####Step 2 : Convert the training dataset to train-ready state.

        python pre_works_train.py
        
####Step 3 : Train the model.

  for a channel training
  
        python create_model.py -a 
  
  for b channel training
  
        python create_model.py -b 
        
####Step 4 : Convert the testing dataset to test-ready state

        python pre_works_test.py
        
####Step 5 : Test the model

  for a channel testing
  
        python test.py -a 
  
  for b channel testing
  
        python test.py -b
        
  for ab channel testing
  
         python test.py -ab
         
######Note:

1: To test the model in ab-mode, test the model in a and b mode first.
       
2: For running the project for first time.

        sh do_first.sh

3: For cleaning temp files and model.

        sh delete_temps.sh