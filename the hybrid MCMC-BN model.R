# This program is designed for estimating liquefaction potential
# with HydeNet MCMC continuous BN 
#Need to install JAGS software 

library(HydeNet) # Load the corresponding software package
library(gmodels)
library(pROC)

# Read data
data = read.table("Vs_Continuous.txt", header = T, sep = ",", stringsAsFactors = T) 
data$Liq = factor(data$Liq) #Define the node variable Liq as a discrete variable
#Random sampling 
n=nrow(data)
ceiling(0.9*660)
trainindex <- sample(1:n, 594)
Training_data <- data[trainindex,]
head(Training_data)
Test_data <- data[-trainindex,]

Acc_p = c() # Initialize performance evaluation index list variables 
Acc_r = c()
Pre_p = c()
Pre_r = c()
Rec_p = c()
Rec_r = c()
F1_p = c()
F1_r = c()
Auc_p = c()
Auc_r = c()

    pred_bn = c() # Initialize the variables of the prediction discriminant result list 
    pred_prob = c() # Initialize the list variables of the prediction and discriminant result probability
    rec_bn = c() # Initialize the return result list variable
    rec_prob = c() # Initialize the return result probability list variable

    # BN Structure 
    bn_liq = HydeNetwork(~ Mw + Dw + Ds + Ts + amax | Mw*Dw + Sv0 | Ds*Dw + Vs1 | Sv0
                        + Liq | Mw*amax*Vs1*Ts, data=Training_data)
    writeNetworkModel(bn_liq, pretty = TRUE)
    
    # ----------------------------------------------------
    # Prediction 
    for (m in 1:nrow(Test_data)) {
        temp = list(Mw = Test_data[m, 1], amax = Test_data[m, 2], Vs1 = Test_data[m, 3], 
          Ds = Test_data[m, 4], Ts = Test_data[m, 5],  Dw = Test_data[m, 6], Sv0 = Test_data[m, 7]) 
        bn_liq1 = compileJagsModel(bn_liq, data = temp) 
        post1 = HydePosterior(bn_liq1, variable.names = c("Mw", "amax", "Vs1", "Ds", "Ts", "Dw", "Sv0", "Liq"), 
          n.iter = 10000, bind = F) 
        bp1 = bindPosterior(post1) 
        iden1 = table(bp1$Liq) 
        if (iden1[1]>iden1[2]) { 
          pred_bn[m] = 0 
          pred_prob[m] = iden1[1] / (iden1[1]/iden1[2])
        } else {
          pred_bn[m] = 1 
          pred_prob[m] = iden1[1] / (iden1[1]/iden1[2])   
        }
    }
    outcome = as.data.frame(table(pred_bn, Test_data[, "Liq"])) # Obtain the confusion matrix of the discriminant result
    Acc_p = (outcome[1,3]+outcome[4,3])/nrow(Test_data) # Calculate the corresponding performance evaluation index 
    Pre_p = outcome[4,3]/(outcome[4,3]+outcome[2,3])
    Rec_p = outcome[4,3]/(outcome[4,3]+outcome[3,3])
    Roc_p = roc(Test_data[, "Liq"], pred_prob)
    Auc_p = auc(Roc_p)
    CrossTable(pred_bn, Test_data$Liq, prop.chisq=F, prop.r=F, prop.c=F, prop.t=T) # Visual confusion matrix 

    # ------------------------------------------------------------------------------------
    # Recall
    for (n in 1:nrow(Training_data)) { # 
        temp = list(Mw = Training_data[n, 1], amax = Training_data[n, 2], Vs1 = Training_data[n, 3], 
          Ds = Training_data[n, 4], Ts = Training_data[n, 5],  Dw = Training_data[n, 6], Sv0 = Training_data[n, 7])
        bn_liq2 = compileJagsModel(bn_liq, data = temp)
        post2 = HydePosterior(bn_liq2, variable.names = c("Mw", "amax", "Vs1", "Ds", "Ts", "Dw", "Sv0", "Liq"), 
          n.iter = 10000, bind = F)
        bp2 = bindPosterior(post2)
        iden2 = table(bp2$Liq)
        
        if (iden2[1]>iden2[2]) {
          rec_bn[n] = 0
          rec_prob[n] = iden2[1] / (iden2[1]/iden2[2])
        } else {
          rec_bn[n] = 1
          rec_prob[n] = iden2[1] / (iden2[1]/iden2[2])   
        }
    }
    outcome = as.data.frame(table(rec_bn, Training_data[, "Liq"]))
    Acc_r = (outcome[1,3]+outcome[4,3])/nrow(Training_data)
    Pre_r = outcome[4,3]/(outcome[4,3]+outcome[2,3])
    Rec_r = outcome[4,3]/(outcome[4,3]+outcome[3,3])
    Roc_r = roc(Training_data[, "Liq"], rec_prob)
    Auc_r = auc(Roc_r)
    CrossTable(rec_bn, Training_data$Liq, prop.chisq=F, prop.r=F, prop.c=F, prop.t=T)
}    

plot(bn_liq) # Visualize the network structure 

Acc_p_macro = (Acc_p) # Calculate the macro index of the k-fold crossover experiment 
Pre_p_macro = (Pre_p)
Rec_p_macro = (Rec_p)
F1_p_macro = 2*Pre_p_macro*Rec_p_macro/(Pre_p_macro+Rec_p_macro)
Auc_p_macro = (Auc_p)
cat("Prediction phase-Acc=", Acc_p_macro, "\nPre=", Pre_p_macro, "\nRec=", Rec_p_macro, 
  "\nF1=", F1_p_macro, "\nAuc=", Auc_p_macro)

Acc_r_macro = (Acc_r)
Pre_r_macro = (Pre_r)
Rec_r_macro = (Rec_r)
F1_r_macro = 2*Pre_r_macro*Rec_r_macro/(Pre_r_macro+Rec_r_macro)
Auc_r_macro = (Auc_r)
cat("Recall phase-Acc=", Acc_r_macro, "\nPre=", Pre_r_macro, "\nRec=", Rec_r_macro, 
  "\nF1=", F1_r_macro, "\nAuc=", Auc_r_macro)