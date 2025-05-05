#-----------------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------#
#
#
#                  Various semi-supervised estimation methods for M-estimation
#          (mainly for linear working model;logistic working model;quantile working model)
#
#-----------------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------#


#################### (1) proposed method (projection-based semi-supervised estimator): PSSE################# 
PSSE <- function(labelled_data,unlabelled_data,c1=NULL,
                 type="linear",tau=0.5,alpha=NULL,gamma=10,sd=FALSE,Kfolds=5)
{
  #-----------------------------------------Arguments-----------------------------------------------------#
  # Purpose: This function is to implement our proposed semi-supervised method. 
  #           This function is the main function, and it also involves several sub-functions defined in
  #           (1.1) - (1.6) below.
  #
  # Input: 
  #      labelled_data: A matrix, whose each row is an observation of predictor vector and the response
  #                      variable and its first column is the observation vector of the response variable.  
  #      unlabelled_data: A matrix, whose each row is an observation of predictor vector. 
  #      c1: Weight. It is the weight to balance the contribution of labelled data and unlabelled data.
  #           Default value is nrow(labelled_data)/(nrow(labelled_data)+nrow(unlabelled_data)).
  #      type: A specification for the type of loss function $L$. We offer three choices: 
  #             "linear", "logistic", "quantile". Default value is "linear". 
  #      tau: The quantile level, which is only useful for "type=quantile". Default value is 0.5.
  #      alpha: The polynomial order. If it is NULL, then we use the data-driven selector GBIC_ppo to 
  #              determine the polynomial order. Default is NULL. 
  #      gamma: The maximum of possible polynomial order. Default is 10. Only if alpha is NULL, gamma will 
  #              be usful for the selection of polynomial order in the construction of Z. 
  #      sd: Return the pointwise standard deviation estimate based on the available samples? Default 
  #           value is FALSE.  
  #      Kfolds: K-folds cross validation method to avoiding the over-fitting during estimating variance, 
  #          which is only useful When "sd=TRUE".
  #
  # Output: 
  #       Hattheta: The estimate for the target parameter. 
  #       sd.of.hattheta: If "sd=TRUE", the elementwise standard deviation estimate for the estimator; 
  #                       otherwise, NULL.
  #       alpha: The selected polynomial order. 
  #------------------------------------------------------------------------------------------------------#
  
  # some basic parameters 
  n=nrow(labelled_data)
  N=nrow(unlabelled_data)
  p=ncol(labelled_data)-1
  

  if (p!=ncol(unlabelled_data)){
    print("the dimension of the labelled and unlabelled data is unmatched")
    
    return(NULL)
  }
  
  if(is.null(c1)==TRUE){
    c1=n/(n+N)
  }
  
  # whether the polynomial order needs to be selected or not 
  if (is.null(alpha)==TRUE){ 
    
    # data splitting 
    labelled_data_part1<- labelled_data[1:round(n/2),]
    labelled_data_part2<- labelled_data[(round(n/2)+1):n,]
    
    
    # the supervised estimate
    hattheta_supervised_part1<- supervised_one(labelled_data_part1,type=type,tau_level=tau)
    hattheta_supervised_part2<- supervised_one(labelled_data_part2,type=type,tau_level=tau)
    
      
    # the first derivative of the loss function L using the supervised estimate 
    L_first_derivative_part1<- L_first_derivative(labelled_data_part1,hattheta_supervised_part2,type=type,tau_level=tau)
    L_first_derivative_part2<- L_first_derivative(labelled_data_part2,hattheta_supervised_part1,type=type,tau_level=tau)
    
    
    # new data integrating the first derivatives of loss function
    Ynew<- rbind(L_first_derivative_part1,L_first_derivative_part2)
    covariates_labelled<- labelled_data[,-1]
    
    
    # GBIC_ppo to select the polynomial order 
    GBICscrores<-apply(as.matrix(1:gamma,1,gamma), 1, function(t) GBIC_ppo(Ynew,covariates_labelled,t))
    alpha<- which.min(GBICscrores)
    
  }
  
  # determine Z
  labelled_Z <- polynomial(labelled_data[,-1],alpha)
  unlabelled_Z <- polynomial(unlabelled_data,alpha)
  
  
  
  # weights (w_i) using in the our optimization problem to get the proposed semi-supervised estimate 
  unlabelled_Z_mean <- colMeans(unlabelled_Z)
  labelled_Z_seondmoment <- t(labelled_Z)%*%labelled_Z/n
  weights_loss <- as.vector(c1+(1-c1)*t(unlabelled_Z_mean)%*%solve(labelled_Z_seondmoment)%*%t(labelled_Z))

  
  
  # estimate of the target parameter
  if (type=="linear"){
    hattheta_PSSE <- as.vector(solve(t(cbind(rep(1,n),labelled_data[,-1]))%*%
                                       (weights_loss*cbind(rep(1,n),labelled_data[,-1])))%*%t(cbind(rep(1,n),labelled_data[,-1]))%*%
                                 (weights_loss*labelled_data[,1]))
  }
  
  if (type=="logistic"){
    hattheta_PSSE <- newton_iteration(labelled_data,tol=10^(-5),max_i=50,weights=weights_loss)
  }
  
  else if(type=="quantile"){
    
    beta_initial<- rq(as.vector(labelled_data[,1]) ~ labelled_data[,-1], tau=tau, weights=weights_loss*(weights_loss>=0), 
                      data = as.data.frame(labelled_data))$coefficients
    if(sum(weights_loss<0)>0)
    {
      tol=10^(-5)
      max_i=100
      rate=10^(-3)
      beta_initial<- gradient_descent(labelled_data,beta_initial=beta_initial,rate=rate,
                                      tol,max_i,weights=weights_loss,type="quantile",tau=tau)
      
    }
    hattheta_PSSE<- beta_initial
  }
  hattheta_PSSE = list("Hattheta"=hattheta_PSSE)
  
  ### estimating the sd
  if(sd==TRUE){
    
    if(type=="linear"){
      
      
      X_secondmoment_inverse=solve(t(cbind(rep(1,N),unlabelled_data))%*%cbind(rep(1,N),unlabelled_data)/N) 
      
      set.seed(20218080)
      index=createFolds(1:n, k = Kfolds) # data splitting
      W1_test = vector()
      
      
      for(k in 1:Kfolds){
        
        index_k=as.vector(index[[k]])
        Yt_train = labelled_data[-index_k,1]
        Xt_train = labelled_data[-index_k,-1]
        Zt_train = labelled_Z[-index_k,]
        Yt_test = labelled_data[index_k,1]
        Xt_test = labelled_data[index_k,-1]
        Zt_test = labelled_Z[index_k,]
        
        nrow_Xt_train = n-length(index_k)
        nrow_Xt_test = length(index_k)
        
        
        
        # projection matrix A(theta) estimation 
        
        hattheta_supervised_train_cv <- hattheta_PSSE[[1]]#lm(Yt_train~Xt_train)$coefficients
        L_firstder_train_cv <- as.vector(Yt_train-cbind(rep(1,nrow_Xt_train),Xt_train)%*%hattheta_supervised_train_cv)*
          cbind(rep(1,nrow_Xt_train),Xt_train)
        L_firstder_projection_cof <- solve(t(Zt_train)%*%Zt_train/nrow_Xt_train)%*%
          t(Zt_train)%*%L_firstder_train_cv/nrow_Xt_train
        
        
        # estimate variance by test data
        
        L_firstder_test_cv <- as.vector(Yt_test-cbind(rep(1,nrow_Xt_test),Xt_test)%*%hattheta_supervised_train_cv)*
          cbind(rep(1,nrow_Xt_test),Xt_test)
        
        W1_test_k <- L_firstder_test_cv+(c1-1)*Zt_test%*%L_firstder_projection_cof
        W1_test<- rbind(W1_test,W1_test_k)
        
        
        
      }
      
      
      
      L_firstder_total <- as.vector(labelled_data[,1]-(cbind(rep(1,n),labelled_data[,-1]))%*%hattheta_PSSE[[1]])*
        (cbind(rep(1,n),labelled_data[,-1]))
      L_firstder_projection_cof_total <- solve(t(labelled_Z)%*%labelled_Z/n)%*%t(labelled_Z)%*%L_firstder_total/n
      W2_total <- (1-c1)*unlabelled_Z%*%L_firstder_projection_cof_total
      
      W1_covariance <- t(W1_test)%*%W1_test/n
      W2_covariance <- t(W2_total)%*%W2_total/N
      Vc_hat_semi = W1_covariance+(n/N)*W2_covariance
      
      Var_matrix_hat = X_secondmoment_inverse%*%Vc_hat_semi%*%X_secondmoment_inverse 
      sd_hattheta_PSSE = sqrt(diag(Var_matrix_hat/n))
    }
    
    if(type=="logistic"){
      
      exp_linear_combined <- as.vector(cbind(rep(1,N),unlabelled_data)%*%hattheta_PSSE[[1]])
      weights_second_derivative <- 1/(1+exp(-exp_linear_combined))^2*exp(-exp_linear_combined)
      X_secondmoment_inverse=solve(t(weights_second_derivative*cbind(rep(1,N),unlabelled_data))%*%cbind(rep(1,N),unlabelled_data)/N) 
      
      set.seed(20218080)
      index=createFolds(1:n, k = Kfolds) # data splitting
      W1_test = vector()
      
      
      for(k in 1:Kfolds){
        
        index_k=as.vector(index[[k]])
        Yt_train = labelled_data[-index_k,1]
        Xt_train = labelled_data[-index_k,-1]
        Zt_train = labelled_Z[-index_k,]
        Yt_test = labelled_data[index_k,1]
        Xt_test = labelled_data[index_k,-1]
        Zt_test = labelled_Z[index_k,]
        
        nrow_Xt_train = n-length(index_k)
        nrow_Xt_test = length(index_k)
        
        
        
        # projection matrix A(theta) estimation 
        
        hattheta_supervised_train_cv <- hattheta_PSSE[[1]]#lm(Yt_train~Xt_train)$coefficients
        exp_linear_combined_train <- as.vector(cbind(rep(1,nrow_Xt_train),Xt_train)%*%hattheta_supervised_train_cv)
        L_firstder_train_cv <- as.vector(1/(1+exp(-exp_linear_combined_train))-Yt_train)*
          cbind(rep(1,nrow_Xt_train),Xt_train)
        L_firstder_projection_cof <- solve(t(Zt_train)%*%Zt_train/nrow_Xt_train)%*%
          t(Zt_train)%*%L_firstder_train_cv/nrow_Xt_train
        
        
        # estimate variance by test data
        
        exp_linear_combined_test <- as.vector(cbind(rep(1,nrow_Xt_test),Xt_test)%*%hattheta_supervised_train_cv)
        L_firstder_test_cv <- as.vector(1/(1+exp(-exp_linear_combined_test))-Yt_test)*
          cbind(rep(1,nrow_Xt_test),Xt_test)
        
        W1_test_k <- L_firstder_test_cv+(c1-1)*Zt_test%*%L_firstder_projection_cof
        W1_test<- rbind(W1_test,W1_test_k)
        
      }
      
      
      exp_linear_combined_total <- as.vector(cbind(rep(1,n),labelled_data[,-1])%*%hattheta_PSSE[[1]])
      L_firstder_total <- (1/(1+exp(-exp_linear_combined_total))-labelled_data[,1])*
        (cbind(rep(1,n),labelled_data[,-1]))
      L_firstder_projection_cof_total <- solve(t(labelled_Z)%*%labelled_Z/n)%*%t(labelled_Z)%*%L_firstder_total/n
      W2_total <- (1-c1)*unlabelled_Z%*%L_firstder_projection_cof_total
      
      W1_covariance <- t(W1_test)%*%W1_test/n
      W2_covariance <- t(W2_total)%*%W2_total/N
      Vc_hat_semi = W1_covariance+(n/N)*W2_covariance
      
      Var_matrix_hat = X_secondmoment_inverse%*%Vc_hat_semi%*%X_secondmoment_inverse 
      sd_hattheta_PSSE = sqrt(diag(Var_matrix_hat/n))
      
    }
    
    if(type=="quantile"){
      
      set.seed(20218080)
      index=createFolds(1:n, k = Kfolds) # data splitting
      W1_test = vector()
      
      
      for(k in 1:Kfolds){
        
        index_k=as.vector(index[[k]])
        Yt_train = labelled_data[-index_k,1]
        Xt_train = labelled_data[-index_k,-1]
        Zt_train = labelled_Z[-index_k,]
        Yt_test = labelled_data[index_k,1]
        Xt_test = labelled_data[index_k,-1]
        Zt_test = labelled_Z[index_k,]
        
        nrow_Xt_train = n-length(index_k)
        nrow_Xt_test = length(index_k)
        
        
        residual_train_indicator <- as.vector(Yt_train-cbind(rep(1,nrow_Xt_train),Xt_train)%*%hattheta_PSSE[[1]])<=0
        L_firstder_train_cv <- (residual_train_indicator-tau)*cbind(rep(1,nrow_Xt_train),Xt_train)
        L_firstder_projection_cof <- solve(t(Zt_train)%*%Zt_train)%*%
          t(Zt_train)%*%L_firstder_train_cv
        
        
        # estimate variance by test data
        
        residual_test_indicator <- as.vector(Yt_test-cbind(rep(1,nrow_Xt_test),Xt_test)%*%hattheta_PSSE[[1]])<=0
        L_firstder_test_cv <- (residual_test_indicator-tau)*cbind(rep(1,nrow_Xt_test),Xt_test)
        
        W1_test_k <- L_firstder_test_cv+(c1-1)*Zt_test%*%L_firstder_projection_cof
        W1_test<- rbind(W1_test,W1_test_k)
        
      }
      
      
      residual_total_indicator <- as.vector(labelled_data[,1]-cbind(rep(1,n),labelled_data[,-1])%*%hattheta_PSSE[[1]])<=0
      L_firstder_total <- (residual_total_indicator-tau)*(cbind(rep(1,n),labelled_data[,-1]))
      L_firstder_projection_cof_total <- solve(t(labelled_Z)%*%labelled_Z)%*%t(labelled_Z)%*%L_firstder_total
      W2_total <- (1-c1)*unlabelled_Z%*%L_firstder_projection_cof_total
      
      W1_covariance <- t(W1_test)%*%W1_test/n
      W2_covariance <- t(W2_total)%*%W2_total/N
      Vc_hat_semi = W1_covariance+(n/N)*W2_covariance
      
      
      
      ##estimating the second derivatives (M)
      B=2000
      G=mvrnorm(B,rep(0,(p+1)),diag(rep(1,(p+1))))
      theta_check_semi <- apply(1/sqrt(n)*G,1,function(t) hattheta_PSSE[[1]]+t)
      residual_semi_indicator= apply(cbind(rep(1,n),labelled_data[,-1])%*%theta_check_semi,2, function(t) as.vector(labelled_data[,1]-t)<=0)
      U_check_semi <- t(residual_semi_indicator-tau)%*%cbind(rep(1,n),labelled_data[,-1])/sqrt(n)
      
      hat_M_semi <- t(apply(U_check_semi,2, function(t) lm(t~G-1)$coefficients))
      
      
      if(class(try(solve(hat_M_semi),silent=T))[1]=="try-error"){
        X_secondmoment_inverse <- solve((hat_M_semi+t(hat_M_semi))/2)
      } else{
        X_secondmoment_inverse<- solve(hat_M_semi)
      }
      
      
      
      
      Var_matrix_hat = X_secondmoment_inverse%*%Vc_hat_semi%*%t(X_secondmoment_inverse)
      sd_hattheta_PSSE = sqrt(diag(Var_matrix_hat)/n)
    }
      
    
    hattheta_PSSE = append(hattheta_PSSE,list("sd.of.hattheta"=sd_hattheta_PSSE))
  }
  
  hattheta_PSSE = append(hattheta_PSSE,list("alpha"=alpha))
  
  return(hattheta_PSSE)
}


###############(1.1) polynomial generation function 

polynomial<- function(data,order) 
{
  #---------------------------Arguments----------------------------------------#
  # Purpose: This function is to produce the polynomial basis of X including 
  #           the intercept vector. 
  #
  # Input: 
  #       data: A matrix, whose each row is an observation of predictor vector.
  #       order: The polynomial order.
  #----------------------------------------------------------------------------#
  polynomial_Z<- rep(1,nrow(data))
  for (i in 1:order)
  {
    polynomial_i <- data^i
    polynomial_Z <- cbind(polynomial_Z,polynomial_i)
  }
  return(polynomial_Z)
}




###############(1.2) the data driven selector GBIC_ppo  
GBIC_ppo <- function (Y_new,covariates_matrix,order_poly){
  #-----------------------Arguments--------------------------------------------------# 
  # Purpose: GBIC_ppo criterion function 
  #
  # Input:
  #       Y_new: The first derivatives of the loss function L.
  #       covariates_matrix: A matrix, each row is an observation of predictor vector.
  #       order_poly: The polynomial order.
  #
  #---------------------------------------------------------------------------------#
  d <- ncol(Y_new)
  n <- nrow(Y_new)
  p <- ncol(covariates_matrix)
  
  polynomial_matrix <- polynomial(covariates_matrix, order_poly)
  
  gammahat <- lm(Y_new~polynomial_matrix-1)$coefficients
  residual_square <- (Y_new - polynomial_matrix %*% gammahat)^2
  sigmahat_square <- colSums(residual_square)/(n-p*order_poly-1)
  design_matrix <- t(polynomial_matrix)%*%polynomial_matrix/n
  
  
  if(class(try(solve(design_matrix),silent=T))[1]=="try-error"){
    
    GBIC<- 9999999
    
  }else {
    trace_det_AB = numeric()
    for (j in 1:d){
      Bhat_j<- t(polynomial_matrix*residual_square[,j])%*%polynomial_matrix/n
      cova_constrast <- 1/sigmahat_square[j]*solve(design_matrix)%*%Bhat_j
      trace_AB <- sum(diag(cova_constrast))
      det_AB <- det(cova_constrast)
      trace_det_AB=c(trace_det_AB,trace_AB-log(det_AB))
    }
    GBIC = 1/n*(d*(n-p*order_poly-1)+ n*sum(log(sigmahat_square)) + d*log(n)*p*order_poly+sum(trace_det_AB))
  }
  
  return(GBIC)
  
}


#################(1.3) the supervised estimate based on the given samples
supervised_one<- function(labelled_data,type=type,tau_level=0.5){
  #---------------------Arguments-------------------------------------------#
  # Purpose: According the type of loss function, this function is used to  
  #         get the estimate for the target parameter only based on the given 
  #         labelled samples.
  #------------------------------------------------------------------------#
  if (type=="linear"){
    hattheta_supervised <- lm(labelled_data[,1]~labelled_data[,-1])$coefficients
    
  }
  
  if (type=="logistic"){
    hattheta_supervised <- glm(labelled_data[,1]~labelled_data[,-1],family="quasibinomial")$coefficients
  
  }
  
  else if (type=="quantile"){
    hattheta_supervised <- rq(labelled_data[,1]~labelled_data[,-1],tau_level)$coefficients
  }
  
  return(hattheta_supervised)
}




#################(1.4)the first derivative of loss function L
L_first_derivative<- function(labelled_data,hattheta_supervised,type,tau_level=0.5)
{
  #---------------------Arguments-------------------------------------------#
  # Purpose: According the type of loss function, this function is used to  
  #         determine the first-order derivatives of the loss function based on 
  #         an estimate of the target parameter and a labelled data set.
  #------------------------------------------------------------------------#
  n=nrow(labelled_data)
  
  if (type=="linear"){
    
    residuals<- as.vector(labelled_data[,1]-cbind(rep(1,n),labelled_data[,-1])%*%hattheta_supervised)
    L_first_derivative <- residuals*cbind(rep(1,n),labelled_data[,-1])
    
  }
  
  if (type=="logistic"){
  
    exp_numerator <- as.vector(exp(cbind(rep(1,n),labelled_data[,-1])%*%hattheta_supervised))
    L_first_derivative <-  (exp_numerator/(1+exp_numerator)-labelled_data[,1])*cbind(rep(1,n),labelled_data[,-1])
  }
  
  else if (type=="quantile"){

    residuals<- as.vector(labelled_data[,1]-cbind(rep(1,n),labelled_data[,-1])%*%hattheta_supervised)
    L_first_derivative <- (tau_level-(residuals<=0))*cbind(rep(1,n),labelled_data[,-1])
  }
  
  return(L_first_derivative)
}



##################(1.5) newton_iteration for logistic model
newton_iteration<-function(data,tol,max_i,weights=rep(1,nrow(data)))
{
  #-----------------------Arguments-----------------------------------------#
  # Purpose: This function aims at estimating the coefficients
  #         by newton iteration method for logistic regression working model.
  #
  # Input: 
  #       tol: The tolerance parameter.
  #       max_i: Maximum number of iterations. 
  #       weights: Observation weights.
  #------------------------------------------------------------------------#
  
  
  #print(c("Newton","theta",theta))
  n=nrow(data)
  p=ncol(data)-1
  X_data=cbind(rep(1,n),data[,-1])
  beta_initial=as.vector(lm(data[,1]~as.matrix(data[,-1]))$coefficients)
  i=0
  beta_distance=1
  while(beta_distance>tol & i<=max_i)
  { 
    linear_combined <- as.vector(X_data%*%beta_initial)
    p_function=1/(1+exp(-linear_combined))
    weight_theta=p_function*(1-p_function)
    # if(sqrt(sum((weight_theta)^2))<tol)
    #    break
    beta_item1=t(X_data)%*%diag(weights)%*%(data[,1]-p_function)
    beta_item2=t(X_data)%*%diag(weights*weight_theta)%*%X_data
    beta_iteration=beta_initial+as.vector(solve(beta_item2)%*%beta_item1)
    beta_distance=sqrt(sum((beta_iteration-beta_initial)^2))
    beta_distance
    beta_initial=beta_iteration
    i=i+1
    #print(c("i",i))
    #print(beta_initial)
  }
  return(beta_initial)
  
}  



##################(1.6)gradient descent for logistic model etc.
gradient_descent<-function(data,beta_initial,rate,tol,max_i,weights=rep(1,nrow(data)),type="logistic",tau=0.5)
{
  #--------------------------------Arguments--------------------------------------------------#
  # Purpose: This function aims at estimating the coefficients by gradient descent method for 
  #           logistic model or using the gradient descent method to fine-tune a given 
  #           estimate of the target parameter for quantile regression working model. 
  #
  # Input:
  #       beta_initial: An initial estimate for the target parameter
  #       rate: Learning rate.
  #       tol: The tolerance parameter in the loop.
  #       max_i: Maximum number of iterations. 
  #       weights: Observation weights.
  #       type: The type of loss function. 
  #-------------------------------------------------------------------------------------------#
  
  n=nrow(data)
  p=ncol(data)-1
  X_data=cbind(rep(1,n),data[,-1])
  i=0
  beta_distance=1
  
  if(type=="logistic"){
    
    while(beta_distance>tol & i<=max_i)
    { 
      p_function = exp(X_data%*%beta_initial)
      p_function=as.vector(p_function/(1+p_function)) 
      
      # large p_function may make the function invalid
      index_inf = which(is.na(p_function)==TRUE)
      p_function[index_inf]=1
      
      weight_theta=p_function*(1-p_function)
      beta_item1=as.vector(-t(weights*X_data)%*%(data[,1]-p_function)/n)
      
      beta_iteration=beta_initial-rate*beta_item1
      beta_distance=sqrt(sum(beta_item1^2))
      #beta_distance
      beta_initial=beta_iteration
      i=i+1
      #print(c("i",i))
      #print(beta_initial)
    }
  }  
  
  if (type=="quantile"){
    
    
    while(beta_distance>tol & i<=max_i)
    { 
      index_tau <- (as.vector(data[,1]-X_data%*%beta_initial)<=0)-tau
      beta_item1=as.vector(-t(weights*X_data)%*%index_tau/n)
      
      beta_iteration=beta_initial-rate*beta_item1
      beta_distance=sqrt(sum((beta_item1)^2))
      beta_initial=beta_iteration
      i=i+1
      
    }
  } 
  
  
  return(beta_initial)
  
}  





###########################(2) PI proposed by Azriel et al. (2021) #################################
PI <- function(labelled_data,unlabelled_data)
{
  #---------------------------------Arguments------------------------------------------#
  # Purpose: This function is to implement the method proposed by Azriel et al.(2021)
  #          for linear working model. 
  #
  # Input:
  #       labelled_data: Same as in the function "PSSE". 
  #       unlabelled_data: Same as in the function "PSSE".     
  #
  # Output: 
  #        Hattheta: The estimate for the target parameter. 
  #------------------------------------------------------------------------------------#
  n=nrow(labelled_data)
  p=ncol(labelled_data)-1
  N=nrow(unlabelled_data)
  
  # combine all the covairates 
  X_combined <- rbind(labelled_data[,-1],unlabelled_data)
  X_labelled <- labelled_data[,-1]
  
  hat_beta_initial <- numeric()
  X_dot <- matrix(rep(0,n*p),n,p) 
  delta_tilde <- matrix(rep(0,n*p),n,p)
  
  for (j in 1:p)
  {
    ## first step 
    coefficients_negtive_j = lm(X_combined[,j]~X_combined[,-j])$coefficients #unlabeled data
    X_j_dot = X_labelled[,j] - cbind(rep(1,n),X_labelled[,-j]) %*% coefficients_negtive_j #labeled data projection error
    X_j_dot_total = X_combined[,j] - cbind(rep(1,n+N),X_combined[,-j]) %*% coefficients_negtive_j
    X_j_dot_square_sampleaverage=mean(X_j_dot_total^2)
    
    
    ## step step 
    W_j = as.vector(labelled_data[,1])*X_j_dot/X_j_dot_square_sampleaverage
    U1 = X_j_dot/X_j_dot_square_sampleaverage
    X_dot[,j] = as.vector(U1) 
    U = matrix(rep(0,n*p),n,p) 
    for (jj in 1:p)
    {
      if (jj==j)
      {
        U[,jj] = X_labelled[,jj]*X_j_dot/X_j_dot_square_sampleaverage-1
      }
      
      else
      {
        U[,jj] = X_labelled[,jj]*X_j_dot/X_j_dot_square_sampleaverage
      }
      
    }
    
    delta_tilde[,j] = W_j-as.matrix(cbind(rep(1,n),U1,U)) %*% (lm(W_j~U1+U)$coefficients)
    hat_beta_j = lm(W_j~U1+U)$coefficients[1]
    hat_beta_initial = c(hat_beta_initial,hat_beta_j)
  }
  hat_alpha = mean(labelled_data[,1])-t(hat_beta_initial)%*%colMeans(X_labelled)
  hat_theta_Azriel = as.vector(c(hat_alpha,hat_beta_initial))
  
  hat_theta_Azriel = list("Hattheta"=hat_theta_Azriel)
  

  
  return(hat_theta_Azriel)
}
#PI(data_labelled,data_unlabelled)



##################(3) EASE proposed by Chakrabortty and Cai (2018) ##########################
EASE<-function(labelled_data,unlabelled_data,K,H,r)
{
  #---------------------------------Arguments-------------------------------------------------#
  # Purpose: This function is to implement the method proposed by Chakrabortty and Cai (2018) 
  #          for linear working model. This is the main function, and it also involves a 
  #          subfunction defined in (3.1). 
  #   
  # Input: 
  #       labelled_data: Same as in the function "PSSE". 
  #       unlabelled_data: Same as in the function "PSSE".
  #       K: K-folds for sliced inverse regression.
  #       H: H slices in sliced inverse regression.
  #       r: The reduced dimension. 
  #
  # Output: 
  #        Hattheta: The estimate for the target parameter. 
  #------------------------------------------------------------------------------------------#
  
  n=nrow(labelled_data)
  p=ncol(labelled_data)-1
  N=nrow(unlabelled_data)
  
  X_combined<- rbind(labelled_data[,-1],unlabelled_data)
  hat_Gamma_semisupervised= t(cbind(rep(1,n+N),X_combined))%*%cbind(rep(1,n+N),X_combined)/(n+N)
  coef_supervised = lm(labelled_data[,1]~labelled_data[,-1])$coefficients
  set.seed(20210907)
  index=createFolds(1:n, k = K) # data splitting
  
  
  #####estimate phi_k(k=1,...,K) and parameters
  hat_nonpara_test_vector <- numeric()
  hat_nonpara_unlabelled_matrix <- vector()
  covariates_test = vector()
  response_test = numeric()
  
  for (k in 1:K)
  {
    
    index_k=as.vector(index[[k]])
    Yt_train = labelled_data[-index_k,1]
    Xt_train = labelled_data[-index_k,-1]
    Yt_test = labelled_data[index_k,1]
    Xt_test = labelled_data[index_k,-1]
    X_combined_train = rbind(Xt_train, unlabelled_data)
    nrow_Xt_train = n-length(index_k)
    nrow_Xt_test = length(index_k)
    
    covariates_test = rbind(covariates_test,Xt_test) #save the covariates for test 
    response_test <- c(response_test,Yt_test)
    
    
    ###SS-STR
    
    ##step(0):estimate mean of covariance of predictors X and standardized 
    X_combined_train_mean = colMeans(X_combined_train)
    X_train_covariance = cov(X_combined_train)
    eigen_X_train_covariance = eigen(X_train_covariance)
    lamda_sqrt <- diag(sqrt(eigen_X_train_covariance$values))
    Sigma_sqrt <- (eigen_X_train_covariance$vectors)%*%(lamda_sqrt*solve(eigen_X_train_covariance$vectors))
    Sigma_negative_sqrt <- solve(Sigma_sqrt)  # calculate the Sigma^{-1/2}
    X_train_standard = t(Sigma_negative_sqrt %*% t(X_combined_train - X_combined_train_mean))
    Xt_train_standard = X_train_standard[1:nrow_Xt_train,]
    Xv_standard = X_train_standard[(nrow_Xt_train+1):(nrow_Xt_train+N),]
    
    
    ##step(i)
    range_Yt_train = range(Yt_train)
    intrval_length = (range_Yt_train[2]-range_Yt_train[1])/H
    indicator_Yt_train = floor(Yt_train/intrval_length)+1 # the interval indicator
    group_with_train = sort(unique(indicator_Yt_train))
    prob_h_k_train = prop.table(table(group_with_train))
    
    
    ##step(ii)
    distance_Xv_Xt_train = matrix(rep(0,N*nrow_Xt_train),N,nrow_Xt_train)
    for(i in 1:N)
    {
      
      for(j in 1:nrow_Xt_train)
      {
        distance_Xv_Xt_train[i,j] = sum((Xv_standard[i,]-Xt_train_standard[j,])^2)
      }
      
    }
    index_Xv_imputed = apply(distance_Xv_Xt_train,1,which.min)
    #index_Xv_tmp_imputed
    indicator_Xv_imputed = indicator_Yt_train[index_Xv_imputed]
    #indicator_Xv_tmp_imputed
    indicator_X_train = c(indicator_Yt_train,indicator_Xv_imputed)
    
    
    ##step(iii)
    M_matrix_train_semi = matrix(rep(0,p*p),p,p)
    for(kk in 1: length(group_with_train))
    {         
      index_train_standard_kk<- which(indicator_X_train==group_with_train[kk])
      if(length(index_train_standard_kk)==1)
      {
        M_bar_sampleaverage = X_train_standard[index_train_standard_kk,]
        M_matrix_train_semi = M_matrix_train_semi + prob_h_k_train[kk]*M_bar_sampleaverage%*%t(M_bar_sampleaverage)
      }
      if (length(index_train_standard_kk)>1)
      {
        M_bar_sampleaverage = colMeans(X_train_standard[index_train_standard_kk,])
        M_matrix_train_semi = M_matrix_train_semi + prob_h_k_train[kk]*M_bar_sampleaverage%*%t(M_bar_sampleaverage)
      }
      
    }
    
    hat_pstar_h_k = eigen(M_matrix_train_semi)$vectors[,1:r]
    hat_Zero_pstar_h_k = Sigma_negative_sqrt %*% hat_pstar_h_k
    
    Xt_train_SIR = Xt_train %*% hat_Zero_pstar_h_k
    Xt_test_SIR = Xt_test %*% hat_Zero_pstar_h_k
    Xt_unlabelled_SIR = unlabelled_data%*%hat_Zero_pstar_h_k
    
    
    ###calculate the nonparametric estimator in test and unlabelled data
    
    h_opt_chark = optimize(bandwidth.cv,c(0.1,2),Y=Yt_train,X=Xt_train_SIR,d=(K-1),order=r)$minimum
    
    
    hat_nonpara_test = rep(0,nrow_Xt_test)
    for (kkk in 1:nrow_Xt_test)
    {
      kernel = apply(Xt_train_SIR,1, function(t){kernel_guassian(sqrt(sum((Xt_test_SIR[kkk,]-t)^2))/h_opt_chark^r)}) 
      kernel_mean = sum(kernel)
      kernel_Y_mean = t(kernel)%*%Yt_train
      hat_nonpara_test[kkk] = kernel_Y_mean/(kernel_mean+10^(-6))
      
    }
    #hat_nonpara_test  #the nonparametric estimator in test data
    
    hat_nonpara_test_vector <- c(hat_nonpara_test_vector,hat_nonpara_test)
    
    
    
    hat_nonpara_unlabelled = rep(0,N)
    for (kkkk in 1:N)
    {
      kernel = apply(Xt_train_SIR,1, function(t){kernel_guassian(sqrt(sum((Xt_unlabelled_SIR[kkkk,]-t)^2))/h_opt_chark^r)}) 
      kernel_mean = sum(kernel)
      kernel_Y_mean = t(kernel)%*%Yt_train
      hat_nonpara_unlabelled[kkkk] = kernel_Y_mean/(kernel_mean+10^(-6))
      
    }
    #hat_nonpara_unlabelled #the nonparametric estimator in unlabelled data
    
    hat_nonpara_unlabelled_matrix<- cbind(hat_nonpara_unlabelled_matrix,hat_nonpara_unlabelled)
    
  }
  
  
  
  ###estimate eta
  
  hat_eta_k_semisupervised = lm((response_test-hat_nonpara_test_vector)~covariates_test)$coefficients
  #hat_eta_k_semisupervised
  
  ###estimate mu for the unlabelled data
  
  hat_mu_unlabelled = rowMeans(hat_nonpara_unlabelled_matrix)+
    as.vector(cbind(rep(1,N),unlabelled_data) %*% hat_eta_k_semisupervised)
  
  
  ###(SNP estimator) estimate theta by unlabelled data
  
  hattheta_K <- lm(hat_mu_unlabelled~unlabelled_data)$coefficients
  
  
  
  ### EASE estimator
  
  
  ##step(i) estimate hat_eta_k, hat_mu_k,hat_phi_k,hat_phi_0
  
  length_intial <- 0
  hat_Gamma_semisupervised_inverse <- solve(hat_Gamma_semisupervised)
  
  hat_phi_k_cv_matrix<- vector()
  hat_phi_0_cv_matrix <- vector()
  for (k in 1:K)
  {
    length_k = length(index[[k]])
    length_intial = length_intial+length_k
    index_test_k = (1+length_intial-length_k):length_intial
    index_exclude_k = (1:n)[-index_test_k]
    hat_eta_k_cv <- lm((response_test[index_exclude_k]-hat_nonpara_test_vector[index_exclude_k])~
                         covariates_test[index_exclude_k,])$coefficients
    
    hat_mu_k_cv <- hat_nonpara_test_vector[index_test_k]+
      as.vector(cbind(rep(1,length_k),covariates_test[index_test_k,])%*%hat_eta_k_cv)
    
    hat_phi_k_cv <- as.vector(response_test[index_test_k]-hat_mu_k_cv)*cbind(rep(1,length_k),
                                                                             covariates_test[index_test_k,])%*% hat_Gamma_semisupervised_inverse
    
    hat_phi_0_cv <- as.vector(response_test[index_test_k]-cbind(rep(1,length_k),
                                                                covariates_test[index_test_k,])%*%coef_supervised)*cbind(rep(1,length_k),
                                                                                                                         covariates_test[index_test_k,])%*% hat_Gamma_semisupervised_inverse
    
    hat_phi_k_cv_matrix <- rbind(hat_phi_k_cv_matrix,hat_phi_k_cv)
    hat_phi_0_cv_matrix <- rbind(hat_phi_0_cv_matrix,hat_phi_0_cv)
    
  }
  
  ##step(ii) calculate delta_lk
  
  hat_delta_12 = colMeans(hat_phi_0_cv_matrix*(hat_phi_k_cv_matrix-hat_phi_0_cv_matrix))
  hat_delta_22 =colMeans((hat_phi_k_cv_matrix-hat_phi_0_cv_matrix)^2)
  epsilon_n = 1/n^(3/8)
  delta_lk = hat_delta_12/(hat_delta_22+epsilon_n)
  
  
  
  ##step(iii) calculate EASE estimator
  
  hat_theta_semi_Chark<- coef_supervised+delta_lk*(hattheta_K-coef_supervised)
  
  
  
  hat_theta_semi_Chark = list("Hattheta"=hat_theta_semi_Chark)
  
  

  
  return(hat_theta_semi_Chark)
}
#EASE(data_labelled,data_unlabelled,K=5,H=40,r=2)[[1]]



#############(3.1) bandwidth selection
kernel_guassian <- function(a) #guassian kernel
{
  1/sqrt(2*pi)*exp(-a^2/2)
}



bandwidth.cv <- function(Y,X,d,h,order) #bandwidth selection
{
  #------------------------Arguments---------------------------------#
  # Purpose: This function is used for the bandwidth selection
  #          in the function "EASE". After SIR, we need to estimate
  #          the $E[Y|P_rX]$ using kernel smoothing. 
  #
  # Input:
  #       Y: Response observation vector.
  #       X: Covariate matrix.
  #       d: d-folds.
  #       h: bandwidth
  #       order: The power.
  #------------------------------------------------------------------#
  
  n = length(Y)
  p = ncol(X)
  
  set.seed(20220907)
  index=createFolds(1:n, k = d)
  
  err_test=0
  for (i in 1:d)
  {
    index_i=as.vector(index[[i]])
    Yt_train = Y[-index_i]
    Xt_train = X[-index_i,]
    Yt_test = Y[index_i]
    Xt_test = X[index_i,]
    n_positive=length(index_i)
    
    
    for(j in 1:n_positive)
    {
      kernel = apply(Xt_train,1, function(t){kernel_guassian(sqrt(sum((Xt_test[j,]-t)^2))/h^order)}) 
      kernel_mean = sum(kernel)
      kernel_Y_mean = t(kernel)%*%Yt_train
      bias_nonpara_estimator = Yt_test[j]-kernel_Y_mean/kernel_mean
      err_test = err_test + bias_nonpara_estimator^2
    }
    
  } 
  return(err_test)
  
}




#################(4) DRESS proposed by Kawakita and Kanamori (2013) ########################

DRESS<- function(labelled_data,unlabelled_data,type="linear",tau=0.5,L,sd=FALSE,Kfolds=5)
{
  #-------------------------Arguments-------------------------------------------#
  # Purpose: This function is to implement the method proposed by
  #         Kawakita and Kanamori (2013) for M-estimation,
  #         which uses the desity-ratio to improve the estimation
  #         efficiency.
  #
  # Input:
  #       labelled_data: Same as in the function "PSSE". 
  #       unlabelled_data: Same as in the function "PSSE".     
  #       type: Same as in the function "PSSE".
  #       tau: Same as in the function "PSSE".
  #        L: The polynomial order when we construct the polynomial function as
  #            base function for density ratio estimation.
  #       sd: Return the pointwise standard deviation estimate based on the available
  #           samples? Default value is FALSE.
  #       Kfolds: Same as in the function "PSSE". 
  #
  # Output: 
  #       Hattheta: The estimate for the target parameter. 
  #       sd.of.hattheta: If "sd=TRUE", the elementwise standard deviation 
  #                        estimate for the estimator; otherwise, NULL. 
  #       error: indicator of whether the parameters in density-ratio is properly
  #              estimated. "TRUE" represents "NOT"; "FALSE" represents "YES".
  #-----------------------------------------------------------------------------#
  
  n=nrow(labelled_data)
  p=ncol(labelled_data)-1
  N=nrow(labelled_data)
  
  
  base_labelled<- polynomial(labelled_data[,-1],L)
  base_unlabelled <- polynomial(unlabelled_data,L)
  alpha_first_derivative_unlabelled <- colMeans(base_unlabelled)
  
  ## estimating alpha (the parameter in density-ratio)
  tol=10^(-5)
  max_i = 50
  
  alpha_initial <- rep(0,L*p+1)
  alpha_distance<-10
  i=1
  error_svd=FALSE
  while(alpha_distance >tol& i<=max_i){
    exponential_phi_labelled <- as.vector(exp(base_labelled%*%alpha_initial))
    exponential_phi_labelled[which(exponential_phi_labelled==Inf)]=10^(-7)
    
    
    alpha_first_derivative <- colMeans(exponential_phi_labelled*base_labelled)-
      alpha_first_derivative_unlabelled

    alpha_second_derivative <- t(exponential_phi_labelled*base_labelled)%*%base_labelled/n
    
    
    if (max(abs(svd(alpha_second_derivative)$d))>99999|min(abs(svd(alpha_second_derivative)$d))<tol)
    {
      error_svd = TRUE
      alpha_initial<- rep(0,L*p+1)
      break
    }

    alpha_new <- as.vector(alpha_initial-solve(alpha_second_derivative)%*%alpha_first_derivative)


    alpha_distance <- sqrt(sum((alpha_new-alpha_initial)^2))
    alpha_initial<- alpha_new
    i=i+1
    #print(i)
  }


  exponential_phi_labelled <- as.vector(exp(base_labelled%*%alpha_initial))
  
  
  ## estimating theta*(the target parameter)
  if (type=="linear"){
    hattheta_DRESS<- lm(labelled_data[,1]~labelled_data[,-1],weights=exponential_phi_labelled)$coefficients
    
  }else if (type=="logistic"){
    hattheta_DRESS <- glm(labelled_data[,1]~labelled_data[,-1],family="quasibinomial",
                          weights=exponential_phi_labelled)$coefficients
  }
  else if (type=="quantile"){
    hattheta_DRESS <- rq(labelled_data[,1]~labelled_data[,-1],tau=tau,weights=exponential_phi_labelled)$coefficients
  }
  
  
  hattheta_DRESS = append(list("Hattheta"=hattheta_DRESS),list("error"=error_svd))
  
  
  ### estimating the sd
  if(sd==TRUE){
    
    if(type=="linear"){
      
      
      X_secondmoment_inverse=solve(t(cbind(rep(1,N),unlabelled_data))%*%cbind(rep(1,N),unlabelled_data)/N) 
      
      set.seed(20218080)
      index=createFolds(1:n, k = Kfolds) # data splitting
      W1_test = vector()
      
      
      for(k in 1:Kfolds){
        
        index_k=as.vector(index[[k]])
        Yt_train = labelled_data[-index_k,1]
        Xt_train = labelled_data[-index_k,-1]
        Zt_train = base_labelled[-index_k,]
        Yt_test = labelled_data[index_k,1]
        Xt_test = labelled_data[index_k,-1]
        Zt_test = base_labelled[index_k,]
        
        nrow_Xt_train = n-length(index_k)
        nrow_Xt_test = length(index_k)
        
        
        
        # projection matrix A(theta) estimation 
        
        hattheta_supervised_train_cv <-  hattheta_DRESS[[1]]#lm(Yt_train~Xt_train)$coefficients
        L_firstder_train_cv <- as.vector(Yt_train-cbind(rep(1,nrow_Xt_train),Xt_train)%*%hattheta_supervised_train_cv)*
          cbind(rep(1,nrow_Xt_train),Xt_train)
        L_firstder_projection_cof <- solve(t(Zt_train)%*%Zt_train/nrow_Xt_train)%*%
          t(Zt_train)%*%L_firstder_train_cv/nrow_Xt_train
        
        
        # estimate variance by test data
        
        L_firstder_test_cv <- as.vector(Yt_test-cbind(rep(1,nrow_Xt_test),Xt_test)%*%hattheta_supervised_train_cv)*
          cbind(rep(1,nrow_Xt_test),Xt_test)
        
        W1_test_k <- L_firstder_test_cv-Zt_test%*%L_firstder_projection_cof
        W1_test<- rbind(W1_test,W1_test_k)
        
        
        
      }
      
      
      
      L_firstder_total <- as.vector(labelled_data[,1]-(cbind(rep(1,n),labelled_data[,-1]))%*%hattheta_DRESS[[1]])*
        (cbind(rep(1,n),labelled_data[,-1]))
      L_firstder_projection_cof_total <- solve(t(base_labelled)%*%base_labelled/n)%*%t(base_labelled)%*%L_firstder_total/n
      W2_total <- base_unlabelled%*%L_firstder_projection_cof_total
      
      W1_covariance <- t(W1_test)%*%W1_test/n
      W2_covariance <- t(W2_total)%*%W2_total/N
      Vc_hat_semi = W1_covariance+(n/N)*W2_covariance
      
      Var_matrix_hat = X_secondmoment_inverse%*%Vc_hat_semi%*%X_secondmoment_inverse 
      sd_hattheta_DRESS = sqrt(diag(Var_matrix_hat/n))
    }
    
    if(type=="logistic"){
      
      exp_linear_combined <- as.vector(cbind(rep(1,N),unlabelled_data)%*%hattheta_DRESS[[1]])
      weights_second_derivative <- 1/(1+exp(-exp_linear_combined))^2*exp(-exp_linear_combined)
      X_secondmoment_inverse=solve(t(weights_second_derivative*cbind(rep(1,N),unlabelled_data))%*%cbind(rep(1,N),unlabelled_data)/N) 
      
      set.seed(20218080)
      index=createFolds(1:n, k = Kfolds) # data splitting
      W1_test = vector()
      
      
      for(k in 1:Kfolds){
        
        index_k=as.vector(index[[k]])
        Yt_train = labelled_data[-index_k,1]
        Xt_train = labelled_data[-index_k,-1]
        Zt_train = base_labelled[-index_k,]
        Yt_test = labelled_data[index_k,1]
        Xt_test = labelled_data[index_k,-1]
        Zt_test = base_labelled[index_k,]
        
        nrow_Xt_train = n-length(index_k)
        nrow_Xt_test = length(index_k)
        
        
        
        # projection matrix A(theta) estimation 
        
        hattheta_supervised_train_cv <- hattheta_DRESS[[1]]#lm(Yt_train~Xt_train)$coefficients
        exp_linear_combined_train <- as.vector(cbind(rep(1,nrow_Xt_train),Xt_train)%*%hattheta_supervised_train_cv)
        L_firstder_train_cv <- as.vector(1/(1+exp(-exp_linear_combined_train))-Yt_train)*
          cbind(rep(1,nrow_Xt_train),Xt_train)
        L_firstder_projection_cof <- solve(t(Zt_train)%*%Zt_train/nrow_Xt_train)%*%
          t(Zt_train)%*%L_firstder_train_cv/nrow_Xt_train
        
        
        # estimate variance by test data
        
        exp_linear_combined_test <- as.vector(cbind(rep(1,nrow_Xt_test),Xt_test)%*%hattheta_supervised_train_cv)
        L_firstder_test_cv <- as.vector(1/(1+exp(-exp_linear_combined_test))-Yt_test)*
          cbind(rep(1,nrow_Xt_test),Xt_test)
        
        W1_test_k <- L_firstder_test_cv-Zt_test%*%L_firstder_projection_cof
        W1_test<- rbind(W1_test,W1_test_k)
        
      }
      
      
      exp_linear_combined_total <- as.vector(cbind(rep(1,n),labelled_data[,-1])%*%hattheta_DRESS[[1]])
      L_firstder_total <- (1/(1+exp(-exp_linear_combined_total))-labelled_data[,1])*
        (cbind(rep(1,n),labelled_data[,-1]))
      L_firstder_projection_cof_total <- solve(t(base_labelled)%*%base_labelled/n)%*%t(base_labelled)%*%L_firstder_total/n
      W2_total <-  base_unlabelled%*%L_firstder_projection_cof_total
      
      W1_covariance <- t(W1_test)%*%W1_test/n
      W2_covariance <- t(W2_total)%*%W2_total/N
      Vc_hat_semi = W1_covariance+(n/N)*W2_covariance
      
      Var_matrix_hat = X_secondmoment_inverse%*%Vc_hat_semi%*%X_secondmoment_inverse 
      sd_hattheta_DRESS = sqrt(diag(Var_matrix_hat/n))
      
    }
    
    if(type=="quantile"){
      
      
      # estimating $V_c$ by K-folds CV
      
      set.seed(20218080)
      index=createFolds(1:n, k = Kfolds) # data splitting
      W1_test = vector()
      
      
      for(k in 1:Kfolds){
        
        index_k=as.vector(index[[k]])
        Yt_train = labelled_data[-index_k,1]
        Xt_train = labelled_data[-index_k,-1]
        Zt_train = base_labelled[-index_k,]
        Yt_test = labelled_data[index_k,1]
        Xt_test = labelled_data[index_k,-1]
        Zt_test = base_labelled[index_k,]
        
        nrow_Xt_train = n-length(index_k)
        nrow_Xt_test = length(index_k)
        
        
        residual_train_indicator <- as.vector(Yt_train-cbind(rep(1,nrow_Xt_train),Xt_train)%*%hattheta_DRESS[[1]])<=0
        L_firstder_train_cv <- (residual_train_indicator-tau)*cbind(rep(1,nrow_Xt_train),Xt_train)
        L_firstder_projection_cof <- solve(t(Zt_train)%*%Zt_train)%*%
          t(Zt_train)%*%L_firstder_train_cv
        
        
        # estimate variance by test data
        
        residual_test_indicator <- as.vector(Yt_test-cbind(rep(1,nrow_Xt_test),Xt_test)%*%hattheta_DRESS[[1]])<=0
        L_firstder_test_cv <- (residual_test_indicator-tau)*cbind(rep(1,nrow_Xt_test),Xt_test)
        
        W1_test_k <- L_firstder_test_cv-Zt_test%*%L_firstder_projection_cof
        W1_test<- rbind(W1_test,W1_test_k)
        
      }
      
      
      residual_total_indicator <- as.vector(labelled_data[,1]-cbind(rep(1,n),labelled_data[,-1])%*%hattheta_DRESS[[1]])<=0
      L_firstder_total <- (residual_total_indicator-tau)*(cbind(rep(1,n),labelled_data[,-1]))
      L_firstder_projection_cof_total <- solve(t(base_labelled)%*%base_labelled)%*%t(base_labelled)%*%L_firstder_total
      W2_total <-  base_unlabelled%*%L_firstder_projection_cof_total
      
      W1_covariance <- t(W1_test)%*%W1_test/n
      W2_covariance <- t(W2_total)%*%W2_total/N
      Vc_hat_semi = W1_covariance+(n/N)*W2_covariance
      
      
      
      ##estimating the second derivatives (M)
      B=2000
      G=mvrnorm(B,rep(0,(p+1)),diag(rep(1,(p+1))))
      theta_check_semi <- apply(1/sqrt(n)*G,1,function(t) hattheta_DRESS[[1]]+t)
      residual_semi_indicator= apply(cbind(rep(1,n),labelled_data[,-1])%*%theta_check_semi,2, function(t) as.vector(labelled_data[,1]-t)<=0)
      U_check_semi <- t(residual_semi_indicator-tau)%*%cbind(rep(1,n),labelled_data[,-1])/sqrt(n)
      
      hat_M_semi <- t(apply(U_check_semi,2, function(t) lm(t~G-1)$coefficients))
      #X_secondmoment_inverse <- solve((hat_M_semi+t(hat_M_semi))/2)
      X_secondmoment_inverse<- solve(hat_M_semi)
      
      
      
      Var_matrix_hat = X_secondmoment_inverse%*%Vc_hat_semi%*%t(X_secondmoment_inverse)
      sd_hattheta_DRESS = sqrt(diag(Var_matrix_hat)/n)
      
      
      
      
    }
    
    hattheta_DRESS <- append(hattheta_DRESS,list("sd.of.hattheta"=sd_hattheta_DRESS))
  }
  

  
  return(hattheta_DRESS)
}






























