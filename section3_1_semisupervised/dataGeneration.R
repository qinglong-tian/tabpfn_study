#---------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#
#
#
#                                   Data generation functions 
#          (mainly for linear working model;logistic working model;quantile working model)
#
#---------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------#
GenerateData<- function(n,N=0,p,option){
  #-----------------------------------------Arguments--------------------------------------------------------#
  # Purpose: This function is to generate the simulated data under different settings both in Section 4.1  
  #           of the paper and Section 3.2 of the supplementary material. 
  #
  # Input: 
  #      n: The labelled sample size.
  #      N: The unlabelled sample size. The default is 0. When "N=0", no unlabelled data are generated. 
  #      p: The dimension of the predictor vector. 
  #      option: The type of data settings. We offer nine choices: "i","ii","iii","W1","W2","W3",S1","S2","S3".
  #               These nine choices correspond to settings (i) - (iii) in Section 4.1 of the paper, 
  #               settings (W1) - (W2), settings (S1) - (S3) in Section 3.2 of the supplementary material 
  #               in sequence. 
  #
  # Output: 
  #       Data.labelled: A matrix, whose each row is an observation of predictor vector and the response
  #                      variable and its first column is the observation vector of the response variable.
  #       Data.unlabelled: A matrix, whose each row is an observation of predictor vector. 
  #----------------------------------------------------------------------------------------------------------#
  #print(paste("NOTE: The data are generated based on setting (",option,").",sep=""))
  if (option=="i")  
  {
    ###parameters in setting (i)
    sigma= 2   # the sd of the error term eta       
    alpha0=1
    alpha1=rep(1,p)
    alpha2=rep(1,p)
    ### data generation including the labelled data set and the unlabelled data set 
    X_labelled=mvrnorm(n,rep(0,p),diag(rep(1,p)))
    eta=rnorm(n,0,sigma)
    Y_labelled=alpha0+X_labelled%*%alpha1+(X_labelled^3-X_labelled^2+exp(X_labelled))%*%alpha2+eta
    data_labelled=cbind(Y_labelled,X_labelled)  # labelled data set 
    if(N!=0){
      data_unlabelled=mvrnorm(N,rep(0,p),diag(rep(1,p)))  # unlabelled data set 
    }
  }else if (option=="ii"){
    ### parameters in setting (ii)
    alpha0=11
    alpha1=rep(1,p)
    alpha2=rep(-1,p)
    ### data generation including the labelled data set and the unlabelled data set 
    X_partition_label=rbinom((n+N),1,0.5)
    X_partition1 = mvrnorm((n+N),rep(-1,p),X_AR1_covmatrix(0.5,p))
    X_partition2 = mvrnorm((n+N),rep(1,p),X_AR1_covmatrix(0.5,p)) 
    X_matrix = X_partition_label*X_partition1+(1-X_partition_label)*X_partition2
    linear_combined <- alpha0+as.vector(X_matrix[1:n,]%*%alpha1+(X_matrix[1:n,])^2%*%alpha2)
    prob=1/(1+exp(-linear_combined))
    Y_labelled=rbinom(n,1,prob)
    data_labelled=cbind(Y_labelled,X_matrix[1:n,]) # labelled data set 
    if(N!=0){
      data_unlabelled= X_matrix[(n+1):(n+N),]  # unlabelled data set 
    }
  }else if (option=="iii"){
    ### parameters in setting (iii)
    alpha0=1
    alpha1=rep(0.5,p)
    alpha2=1
    alpha3=rep(0.5,round(p/2))
    sigma=1
    ### data generation including the labelled data set and the unlabelled data set 
    X_labelled=mvrnorm(n,rep(0,p),diag(rep(1,p)))
    eta=rnorm(n,0,sigma)
    epsilon <- (1+X_labelled[,(1:round(p/2))]%*%alpha3)*eta
    Y_labelled=alpha0+X_labelled%*%alpha1+alpha2*apply(X_labelled,1, function(x) sum(t(x)%*%x))+epsilon
    data_labelled=cbind(Y_labelled,X_labelled)  # labelled data set 
    if(N!=0){
      data_unlabelled=mvrnorm(N,rep(0,p),diag(rep(1,p)))  # unlabelled data set
    }
  }else if (option=="W1"){
    ### parameters in setting (W1)
    alpha0=1
    alpha1=rep(1,p)
    alpha2=rep(1,p)
    sigma=2
    ### data generation including the labelled data set and the unlabelled data set 
    X_labelled=mvrnorm(n,rep(0,p),diag(rep(1,p)))
    eta=rnorm(n,0,sigma)
    Y_labelled=alpha0+X_labelled%*%alpha1+(X_labelled^3-X_labelled^2+exp(X_labelled))%*%alpha2+eta
    X_labelled_trans = cbind(X_labelled[,1:2],X_labelled[,3:p]^3)
    data_labelled=cbind(Y_labelled,X_labelled_trans) # labelled data set
    if(N!=0){
      X_unlabelled=mvrnorm(N,rep(0,p),diag(rep(1,p)))
      data_unlabelled=cbind(X_unlabelled[,1:2],X_unlabelled[,3:p]^3) # unlabelled data set
    }
  }else if (option=="W2"){
    ### parameters in setting (W2)
    alpha0=11
    alpha1=rep(1,p)
    alpha2=rep(-1,p)
    ### data generation including the labelled data set and the unlabelled data set 
    X_partition_label=rbinom((n+N),1,0.5)
    X_partition1 = mvrnorm((n+N),rep(-1,p),X_AR1_covmatrix(0.5,p))
    X_partition2 = mvrnorm((n+N),rep(1,p),X_AR1_covmatrix(0.5,p)) 
    X_matrix = X_partition_label*X_partition1+(1-X_partition_label)*X_partition2
    linear_combined <- alpha0+as.vector(X_matrix[1:n,]%*%alpha1+(X_matrix[1:n,])^2%*%alpha2)
    prob=1/(1+exp(-linear_combined))
    Y_labelled=rbinom(n,1,prob)
    U_labelled<- X_matrix[1:n,]+sin(X_matrix[1:n,])
    data_labelled=cbind(Y_labelled,U_labelled) # labelled data set 
    if(N!=0){
      data_unlabelled= X_matrix[(n+1):(n+N),]+sin(X_matrix[(n+1):(n+N),])  # unlabelled data set 
    }
  }else if (option=="W3"){
    ### parameters in setting (W3)
    alpha0=1
    alpha1=rep(0.5,p)
    alpha2=1
    alpha3=rep(0.5,round(p/2))
    sigma=1
    ### data generation including the labelled data set and the unlabelled data set 
    X_labelled=mvrnorm(n,rep(0,p),diag(rep(1,p)))
    eta=rnorm(n,0,sigma)
    epsilon <- (1+X_labelled[,(1:round(p/2))]%*%alpha3)*eta
    Y_labelled=alpha0+X_labelled%*%alpha1+alpha2*apply(X_labelled,1, function(x) sum(t(x)%*%x))+epsilon
    if(p==4){
      
      U_labelled= cbind(X_labelled[,1:floor((p+1)/2)],X_labelled[,(floor((p+1)/2)+1):p]^2,X_labelled[,1]*X_labelled[,2])
      if(N!=0){
        X_unlabelled=mvrnorm(N,rep(0,p),diag(rep(1,p)))
        data_unlabelled= cbind(X_unlabelled[,1:floor((p+1)/2)],X_unlabelled[,(floor((p+1)/2)+1):p]^2,
            X_unlabelled[,1]*X_unlabelled[,2])
      }
      
    }else if (p>4){
      
      U_labelled= cbind(X_labelled[,1:floor((p+1)/2)],X_labelled[,(floor((p+1)/2)+1):p]^2,
                                                  X_labelled[,1]*X_labelled[,2],X_labelled[,3]*X_labelled[,4])
      if(N!=0){
        X_unlabelled=mvrnorm(N,rep(0,p),diag(rep(1,p)))
        data_unlabelled=cbind(X_unlabelled[,1:floor((p+1)/2)],X_unlabelled[,(floor((p+1)/2)+1):p]^2,
                                       X_unlabelled[,1]*X_unlabelled[,2],X_unlabelled[,3]*X_unlabelled[,4])
      }
    }
    
    data_labelled=cbind(Y_labelled,U_labelled)  # labelled data set 
  }else if (option=="S1"){
    ###parameters in setting (S1)
    sigma=1 # the variance of the error term 
    ### data generation including the labelled data set and the unlabelled data set 
    X_labelled=mvrnorm(n,rep(0,p),X_AR1_covmatrix(0.5,p))
    eta=rnorm(n,0,sigma)
    Y_labelled=rowMeans(g_S1(X_labelled,eta))+eta
    data_labelled=cbind(Y_labelled,X_labelled) # labelled data set 
    if(N!=0){
      data_unlabelled=mvrnorm(N,rep(0,p),X_AR1_covmatrix(0.5,p)) # unlabelled data set 
    }
  }else if (option=="S2"){
    ### data generation including the labelled data set and the unlabelled data set 
    X_partition_label=rbinom((n+N),1,0.5)
    X_partition1 = mvrnorm((n+N),rep(-1,p),X_AR1_covmatrix(0.8,p))
    X_partition2 = mvrnorm((n+N),rep(1,p),X_AR1_covmatrix(0.8,p)) 
    X_matrix = X_partition_label*X_partition1+(1-X_partition_label)*X_partition2
    linear_combined <- rowSums(g_S2(X_matrix[1:n,]))
    prob=1/(1+exp(-linear_combined))
    Y_labelled=rbinom(n,1,prob) 
    data_labelled=cbind(Y_labelled,X_matrix[1:n,])  # labelled data set 
    if(N!=0){
      data_unlabelled = X_matrix[(n+1):(n+N),] # unlabelled data set
    }
  }else if (option=="S3"){
    ###parameters in setting (S3)
    alpha3=rep(0.5,round(p/2))
    sigma=1
    ### data generation including the labelled data set and the unlabelled data set 
    X_labelled=mvrnorm(n,rep(0,p),diag(rep(1,p)))
    eta=rnorm(n,0,sigma)
    epsilon <- (1+X_labelled[,(1:round(p/2))]%*%alpha3)*eta
    Y_labelled=rowMeans(g_S3(X_labelled,eta))+apply(X_labelled,1, function(x) sum(t(x)%*%x))+epsilon
    data_labelled=cbind(Y_labelled,X_labelled)
    if(N!=0){
      data_unlabelled=mvrnorm(N,rep(0,p),diag(rep(1,p)))
    }
  }
  
  # give the colnames of labelled data set and unlabelled data set 
  dimU<- ncol(data_labelled)-1
  colnames(data_labelled)<- c("Y",paste("U",1:dimU,sep=""))
  res_list<- list("Data.labelled"=data_labelled)
  if (N!=0){
    colnames(data_unlabelled)<- paste("U",1:dimU,sep="")
    res_list<- append(res_list,list("Data.unlabelled"=data_unlabelled))
  }
  return(res_list)
}
###############The covariance matrix of the predictor vector ###################
X_AR1_covmatrix<- function(rho,p){
  #---------------------------Arguments----------------------------------------#
  # Purpose: This function is to produce the covariance matrix of the predictors.
  #
  # Input:
  #      rho: A parameter
  #      p: The dimension of the predictor vector. 
  #----------------------------------------------------------------------------#
  x_covmatrix<- matrix(rep(0,p^2),p,p)
  for (ii in 1:p)
  {
    for (jj in 1:p){
      x_covmatrix[ii,jj]<- rho^(abs(ii-jj))
    }
  }
  return(x_covmatrix)
}

#####################A function related to setting (S1) #########################
g_S1<- function(u,eta)
{
  exp(u+2*cos(u)-0.3*eta)+0.1*exp(u+sin(u))+2.5*u^3 # more complicated with eta 
  
}
#####################A function related to setting (S2) #########################
g_S2<-function(u){
  0.9*u^4*(sin(u))^6-2*u^2+2.2    
}
#####################A function related to setting (S3) #########################
g_S3<- function(u,eta)
{
  0.5*exp(u+2*cos(u)-0.3*eta)+0.6*exp(u+sin(u))+u^3
}

