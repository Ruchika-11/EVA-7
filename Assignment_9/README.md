# **ResNets and Higher Receptive Fields**

### **AIM**

- ✅ Write a custom ResNet architecture for CIFAR10 that has the following architecture:     


```
PrepLayer - Conv 3x3 (s1, p1) >> BN >> RELU [64k]
Layer1 -
X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
Add(X, R1)
Layer 2 -
Conv 3x3 [256k]
MaxPooling2D
BN
ReLU
Layer 3 -
X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
Add(X, R2)
MaxPooling with Kernel Size 4
FC Layer 
SoftMax
```

- ✅ Uses One Cycle Policy such that:
  - Total Epochs = 24
  - Max at Epoch = 5
  - LRMIN = FIND
  - LRMAX = FIND
  - NO Annihilation
- ✅ Uses this transform - RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)
- ✅ Batch size = 512
- ✅ Target Accuracy: 90%.
- ✅ Code should be modular
      

### **MODEL SUMMARY**


```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           1,728
       BatchNorm2d-2           [-1, 64, 32, 32]             128
              ReLU-3           [-1, 64, 32, 32]               0
            Conv2d-4          [-1, 128, 32, 32]          73,728
         MaxPool2d-5          [-1, 128, 16, 16]               0
       BatchNorm2d-6          [-1, 128, 16, 16]             256
              ReLU-7          [-1, 128, 16, 16]               0
            Conv2d-8          [-1, 128, 16, 16]         147,456
       BatchNorm2d-9          [-1, 128, 16, 16]             256
             ReLU-10          [-1, 128, 16, 16]               0
           Conv2d-11          [-1, 128, 16, 16]         147,456
      BatchNorm2d-12          [-1, 128, 16, 16]             256
             ReLU-13          [-1, 128, 16, 16]               0
           Conv2d-14          [-1, 256, 16, 16]         294,912
        MaxPool2d-15            [-1, 256, 8, 8]               0
      BatchNorm2d-16            [-1, 256, 8, 8]             512
             ReLU-17            [-1, 256, 8, 8]               0
           Conv2d-18            [-1, 512, 8, 8]       1,179,648
        MaxPool2d-19            [-1, 512, 4, 4]               0
      BatchNorm2d-20            [-1, 512, 4, 4]           1,024
             ReLU-21            [-1, 512, 4, 4]               0
           Conv2d-22            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-23            [-1, 512, 4, 4]           1,024
             ReLU-24            [-1, 512, 4, 4]               0
           Conv2d-25            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-26            [-1, 512, 4, 4]           1,024
             ReLU-27            [-1, 512, 4, 4]               0
        MaxPool2d-28            [-1, 512, 1, 1]               0
           Linear-29                   [-1, 10]           5,120
================================================================
Total params: 6,573,120
Trainable params: 6,573,120
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 6.44
Params size (MB): 25.07
Estimated Total Size (MB): 31.53
----------------------------------------------------------------
```

### **TRAINING LOGS**

```
EPOCH: 1 LR: 0.004624999999999997
Loss=1.530872106552124 Batch_id=97 Accuracy=31.67: 100%|██████████| 98/98 [01:05<00:00,  1.50it/s]
Test set: Average loss: 0.0030, Accuracy: 4410/10000 (44.10%)

EPOCH: 2 LR: 0.007728773790896266
Loss=1.3288702964782715 Batch_id=97 Accuracy=50.10: 100%|██████████| 98/98 [01:05<00:00,  1.50it/s]
Test set: Average loss: 0.0022, Accuracy: 6037/10000 (60.37%)

EPOCH: 3 LR: 0.015849866685431615
Loss=1.1006373167037964 Batch_id=97 Accuracy=58.52: 100%|██████████| 98/98 [01:05<00:00,  1.51it/s]
Test set: Average loss: 0.0025, Accuracy: 6461/10000 (64.61%)

EPOCH: 4 LR: 0.02587401949849496
Loss=0.9372355937957764 Batch_id=97 Accuracy=65.66: 100%|██████████| 98/98 [01:05<00:00,  1.50it/s]
Test set: Average loss: 0.0016, Accuracy: 7212/10000 (72.12%)

EPOCH: 5 LR: 0.033957191732526454
Loss=0.9079457521438599 Batch_id=97 Accuracy=71.11: 100%|██████████| 98/98 [01:05<00:00,  1.50it/s]
Test set: Average loss: 0.0014, Accuracy: 7542/10000 (75.42%)

EPOCH: 6 LR: 0.036999973668439774
Loss=0.7062349915504456 Batch_id=97 Accuracy=75.65: 100%|██████████| 98/98 [01:05<00:00,  1.50it/s]
Test set: Average loss: 0.0012, Accuracy: 7973/10000 (79.73%)

EPOCH: 7 LR: 0.036742523794325105
Loss=0.5868841409683228 Batch_id=97 Accuracy=78.71: 100%|██████████| 98/98 [01:05<00:00,  1.50it/s]
Test set: Average loss: 0.0012, Accuracy: 7922/10000 (79.22%)

EPOCH: 8 LR: 0.03598747173370877
Loss=0.5253560543060303 Batch_id=97 Accuracy=81.14: 100%|██████████| 98/98 [01:05<00:00,  1.50it/s]
Test set: Average loss: 0.0013, Accuracy: 7868/10000 (78.68%)

EPOCH: 9 LR: 0.03475541333853057
Loss=0.4282555878162384 Batch_id=97 Accuracy=83.04: 100%|██████████| 98/98 [01:05<00:00,  1.50it/s]
Test set: Average loss: 0.0010, Accuracy: 8269/10000 (82.69%)

EPOCH: 10 LR: 0.03307995595007442
Loss=0.5135000944137573 Batch_id=97 Accuracy=83.83: 100%|██████████| 98/98 [01:05<00:00,  1.50it/s]
Test set: Average loss: 0.0012, Accuracy: 8082/10000 (80.82%)

EPOCH: 11 LR: 0.031006801678305985
Loss=0.43410223722457886 Batch_id=97 Accuracy=85.56: 100%|██████████| 98/98 [01:05<00:00,  1.50it/s]
Test set: Average loss: 0.0011, Accuracy: 8279/10000 (82.79%)

EPOCH: 12 LR: 0.028592500767449243
Loss=0.33271220326423645 Batch_id=97 Accuracy=86.69: 100%|██████████| 98/98 [01:05<00:00,  1.49it/s]
Test set: Average loss: 0.0009, Accuracy: 8400/10000 (84.00%)

EPOCH: 13 LR: 0.025902909052739613
Loss=0.4588501453399658 Batch_id=97 Accuracy=87.98: 100%|██████████| 98/98 [01:05<00:00,  1.50it/s]
Test set: Average loss: 0.0010, Accuracy: 8378/10000 (83.78%)

EPOCH: 14 LR: 0.02301139158491205
Loss=0.3296332359313965 Batch_id=97 Accuracy=89.07: 100%|██████████| 98/98 [01:05<00:00,  1.50it/s]
Test set: Average loss: 0.0008, Accuracy: 8649/10000 (86.49%)

EPOCH: 15 LR: 0.01999682142286541
Loss=0.335574209690094 Batch_id=97 Accuracy=90.16: 100%|██████████| 98/98 [01:05<00:00,  1.49it/s]
Test set: Average loss: 0.0008, Accuracy: 8644/10000 (86.44%)

EPOCH: 16 LR: 0.016941428182222416
Loss=0.2035992294549942 Batch_id=97 Accuracy=91.47: 100%|██████████| 98/98 [01:05<00:00,  1.49it/s]
Test set: Average loss: 0.0007, Accuracy: 8822/10000 (88.22%)

EPOCH: 17 LR: 0.013928555025772069
Loss=0.23934228718280792 Batch_id=97 Accuracy=92.28: 100%|██████████| 98/98 [01:05<00:00,  1.49it/s]
Test set: Average loss: 0.0007, Accuracy: 8820/10000 (88.20%)

EPOCH: 18 LR: 0.011040385279248182
Loss=0.17508341372013092 Batch_id=97 Accuracy=93.51: 100%|██████████| 98/98 [01:05<00:00,  1.49it/s]
Test set: Average loss: 0.0007, Accuracy: 8764/10000 (87.64%)

EPOCH: 19 LR: 0.008355700684439308
Loss=0.15654341876506805 Batch_id=97 Accuracy=94.32: 100%|██████████| 98/98 [01:05<00:00,  1.50it/s]
Test set: Average loss: 0.0006, Accuracy: 8954/10000 (89.54%)

EPOCH: 20 LR: 0.0059477324386414294
Loss=0.1270587146282196 Batch_id=97 Accuracy=95.56: 100%|██████████| 98/98 [01:05<00:00,  1.49it/s]
Test set: Average loss: 0.0005, Accuracy: 9107/10000 (91.07%)

EPOCH: 21 LR: 0.003882163638495153
Loss=0.15649941563606262 Batch_id=97 Accuracy=96.69: 100%|██████████| 98/98 [01:05<00:00,  1.49it/s]
Test set: Average loss: 0.0005, Accuracy: 9183/10000 (91.83%)

EPOCH: 22 LR: 0.002215337616332496
Loss=0.11126405000686646 Batch_id=97 Accuracy=97.36: 100%|██████████| 98/98 [01:05<00:00,  1.49it/s]
Test set: Average loss: 0.0005, Accuracy: 9200/10000 (92.00%)

EPOCH: 23 LR: 0.0009927210409468926
Loss=0.08400893956422806 Batch_id=97 Accuracy=98.07: 100%|██████████| 98/98 [01:05<00:00,  1.49it/s]
Test set: Average loss: 0.0005, Accuracy: 9221/10000 (92.21%)

EPOCH: 24 LR: 0.00024766370539132344
Loss=0.07467161118984222 Batch_id=97 Accuracy=98.21: 100%|██████████| 98/98 [01:05<00:00,  1.50it/s]
Test set: Average loss: 0.0005, Accuracy: 9230/10000 (92.30%)
```
### **Graphs**      
![accuracy_graph.PNG](images\accuracy_graph.PNG)       

### **One Cycle Policy Implementation**        
![OCPI.PNG](images\OCPI.PNG)  


### **Misclassified Images**      

Below are examples of some missclassified images in the test set:       
![misclassified_images.PNG](images\misclassified_images.PNG)  

### **Grad-CAM on Misclassified images**   

Below are 10 grad-cam example images for misclasified images:        
![grad1.PNG](images\grad1.PNG)  
![grad2.PNG](images\grad2.PNG)  
![grad3.PNG](images\grad3.PNG)  
![grad4.PNG](images\grad4.PNG)  
![grad5.PNG](images\grad5.PNG)  
