# ***1st Attempt :-***
## **SETUP Code**

### ***Target:***
Getting the data; setting Transforms, Data Loader and Training and Testing loop


### ***Results:***
Parameters: 6.3M   
Best Training Accuracy: 99.94%        
Best Test Accuracy: 99.33%        

### ***Analysis:***
Model with extremely large number of parameters.
It has over-fitting as well as is not able to meet our requirement of 99.4% accuracy.

[Colab Link](https://github.com/Ruchika-11/EVA-7/blob/main/Assignment_5/Session5colab1.ipynb)


---

# ***2nd Attempt :-***

## **Skeleton Code with Less Parameters**

### ***Target:***        
Get the basic skeleton right. Without introducing much changes, We will try to make the model lighter and not add any extra stuff.


### ***Results:***          
Parameters: 15,940      
Best Train Accuracy: 99.09%           
Best Test Accuracy: 98.82%           

### ***Analysis:***         
The model has some over-fitting but is much lighter now. Model can be pushed further to achieve the required accuracy.

[Colab Link](https://github.com/Ruchika-11/EVA-7/blob/main/Assignment_5/Session5colab2.ipynb)


---


# ***3rd Attempt :-*** 

## **REGULARIZATION**

### ***Target:***        
Add Batch-normalization and dropout to increase model efficiency.


### ***Results:***          
Parameters: 16,192        (Batch-Normalization adds some parameters)     
Best Train Accuracy: 99.25%           
Best Test Accuracy: 99.41%           

### ***Analysis:***         
Regularization is working properly. 
We don't see any over-fitting and the model is able to achieve the accuracy of 99.4% but it needs to be a little more stable.

[Colab Link](https://github.com/Ruchika-11/EVA-7/blob/main/Assignment_5/Session5colab3.ipynb)

---

# ***4th Attempt :-***

## **Adding GAP layer to further reduce the number of parameters** 

### ***Target:***  
We'll now add GAP layer to further reduce the number of parameters and also stabilize the model. 
### ***Results:***  
Parameters: 8,790     
Best Train Accuracy: 98.80%       
Best Test Accuracy: 99.13%        
### ***Analysis:***  
No overfitting is seen but accuracy has come down because of the reduction in the number of parameters. We need to push the model a little to get the required accuracy.

[Colab Link](https://github.com/Ruchika-11/EVA-7/blob/main/Assignment_5/Session5colab4.ipynb)

---

# ***5th Attempt :-***

## **Addidng Data Augmentation(Rotation) and LR Scheduler** 

### ***Target:*** 
We'll add one of the Data Augmentation techniques like rotation to stabilize the accuracy. We'll also play around with the learning rates to see what difference it creates. 
 
### ***Results:*** 
Parameters: 8,790      
Best Train Accuracy: 99.26%          
Best Test Accuracy: 99.34%    

### ***Analysis:*** 
No overfitting is seen. In Fact, the model is underfitting now.    
Removed dropout since it was not able to help the model with stability.            
Adding "rotation" and "reducing the learning rate after 6th epoch" has helped in increasing the model's accuracy as well as has made it stable. The model has almost reached the desired level of accuracy.

[Colab Link](https://github.com/Ruchika-11/EVA-7/blob/main/Assignment_5/Session5colab5.ipynb)

