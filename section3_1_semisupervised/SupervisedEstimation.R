#---------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#
#
#
#                                   Supervised coefficients estimation functions
#          (mainly for linear working model;logistic working model;quantile working model)
#
#---------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#


SupervisedEst<- function(Data,option,tau=0.5){
  #-----------------------------------------Arguments--------------------------------------------------------#
  # Purpose: This function is to calculate the supervised coefficients estimate based on the given labelled data. 
  #
  # Input: 
  #      Data: A matrix, whose each row is an observation of the response and the predictor vector and first
  #              column is the observation vector of the response. 
  #      option: The type of data settings. We offer nine choices: "i","ii","iii","W1","W2","W3",S1","S2","S3".
  #               These nine choices correspond to settings (i) - (iii) in Section 4.1 of the paper, 
  #               settings (W1) - (W2), settings (S1) - (S3) in Section 3.2 of the supplementary material 
  #               in sequence. 
  #      tau: For "option" equal to "iii", "W3" or "S3", the quantile level; otherwise, this quantity is useless. 
  #
  # Output: 
  #       Est.coef: the estimated value of the coefficients based on the given data 
  #----------------------------------------------------------------------------------------------------------#
  
  n=nrow(Data)
  if (n>500){
    print(paste("NOTE: The coefficients estimate is computed only using labelled data of size ",n,".",sep=""))
  }
  
  
  if (option=="i")  
  {
    Target<- lm(as.vector(Data[,1])~Data[,-1],data=data.frame(Data))$coefficients
    
    
    
  }else if (option=="ii"){
    
    Target<- glm(as.vector(Data[,1])~Data[,-1],family ="quasibinomial")$coefficients
    
  }else if (option=="iii"){
    
    Target<- rq(as.vector(Data[,1])~Data[,-1], tau=tau, data = as.data.frame(Data))$coefficients
    
  }else if (option=="W1"){
   
    Target<- lm(as.vector(Data[,1])~Data[,-1],data=data.frame(Data))$coefficients
    
  }else if (option=="W2"){
   
    Target<- glm(as.vector(Data[,1])~Data[,-1],family ="quasibinomial")$coefficients
    
  }else if (option=="W3"){
    
    Target<- rq(as.vector(Data[,1])~Data[,-1], tau=tau, data = as.data.frame(Data))$coefficients
    
  }else if (option=="S1"){

    Target<- lm(as.vector(Data[,1])~Data[,-1],data=data.frame(Data))$coefficients
    
  }else if (option=="S2"){
    
    Target<- glm(as.vector(Data[,1])~Data[,-1],family ="quasibinomial")$coefficients
    
  }else if (option=="S3"){
    
    Target<- rq(as.vector(Data[,1])~Data[,-1], tau=tau, data = as.data.frame(Data))$coefficients
    
  }
  
  return(list("Est.coef"=as.vector(Target)))
  
  
}
