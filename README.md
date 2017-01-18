#Repository to test out creating a custom model for TensorFlow

###Step 1 : Convert the training dataset to train-ready state.

        python pre_works_train.py
        
###Step 2 : Train the model.

  for a channel training
  
        python create_model.py -a 
  
  for b channel training
  
        python create_model.py -b 
        
###Step 3: Convert the testing dataset to test-ready state

        python pre_works_test.py
        
###Step 4: Test the model

  for a channel testing
  
        python test.py -a 
  
  for b channel testing
  
        python test.py -b
        
  for ab channel testing
  
         python test.py -ab
         
######Note:

To test the model in ab-mode, test the model in a and b mode first.
       
