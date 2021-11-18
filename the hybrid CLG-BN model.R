# Program for learning Hybrid BN for liquefaction identification
library(bnlearn) # Load related software packages 
library(Rgraphviz)
library(gmodels)
library(pROC)

# Read data
data = read.table("Vs_Continuous.txt", header = T, sep = ",", stringsAsFactors = T) 
data$Liq = factor(data$Liq) # Define the node variable Liq as a categorical variable
#Random sampling 
n=nrow(data)
ceiling(0.9*660)
trainindex <- sample(1:n, 594)
Training_data <- data[trainindex,]
head(Training_data)
Test_data <- data[-trainindex,]
# Structure learning
bn_ini = model2network("[Mw][Ds][Ts][Dw][amax|Mw:Dw][Sv0|Ds:Dw][Vs1|Sv0][Liq|Mw:amax:Vs1:Ts]") 
graphviz.plot(bn_ini, shape = "rectangle") #Visualize the initial network structure  

# Revise structure with expert knowledge 
bn_liq = reverse.arc(bn_ini, from = "Vs1", to = "Liq", check.cycles = T, check.illegal = T) 
bn_liq = reverse.arc(bn_liq, from = "Ts", to = "Liq", check.cycles = T, check.illegal = T)
bn_liq = drop.arc(bn_liq, from = "Mw", to = "Liq")
bn_liq = reverse.arc(bn_liq, from = "amax", to = "Liq", check.cycles = T, check.illegal = T)
bn_liq = set.arc(bn_liq, from = "Liq", to = "Mw", check.cycles = T, check.illegal = T)
graphviz.plot(bn_liq, shape = "rectangle") 

Acc_p = c() #Initialize performance evaluation index list variables
Acc_r = c()
Pre_p = c()
Pre_r = c()
Rec_p = c()
Rec_r = c()
F1_p = c()
F1_r = c()
Auc_p = c()
Auc_r = c()

	# Parameter learning 
	bn_parameter <- bn.fit(bn_liq, Training_data)

	# Predict 
	pred_bn = predict(object = bn_parameter, node = "Liq", data = Test_data, method = "bayes-lw") 
	outcome = as.data.frame(table(pred_bn, Test_data[, "Liq"])) 
	Acc_p[i] = (outcome[1,3]+outcome[4,3])/nrow(Test_data) 
	Pre_p[i] = outcome[4,3]/(outcome[4,3]+outcome[2,3])
	Rec_p[i] = outcome[4,3]/(outcome[4,3]+outcome[3,3])
	#Roc_p = roc(Test_data[, "Liq"], pred_bn) 
	#Auc_p[i] = auc(Roc_p)
 	CrossTable(pred_bn, Test_data$Liq, prop.chisq=F, prop.r=F, prop.c=F, prop.t=T) 

 	# Recall 
	pred_bn = predict(object = bn_parameter, node = "Liq", data = Training_data, method = "bayes-lw") 
	outcome = as.data.frame(table(pred_bn, Training_data[, "Liq"]))
	Acc_r[i] = (outcome[1,3]+outcome[4,3])/nrow(Training_data)
	Pre_r[i] = outcome[4,3]/(outcome[4,3]+outcome[2,3])
	Rec_r[i] = outcome[4,3]/(outcome[4,3]+outcome[3,3])
	#Roc_r = roc(Training_data[, "Liq"], pred_bn)
	#Auc_r[i] = auc(Roc_r)
 	CrossTable(pred_bn, Training_data$Liq, prop.chisq=F, prop.r=F, prop.c=F, prop.t=T)

# Calculate the macro index of the experiment 
Acc_p_macro = (Acc_p) 
Pre_p_macro = (Pre_p)
Rec_p_macro = (Rec_p)
F1_p_macro = 2*Pre_p_macro*Rec_p_macro/(Pre_p_macro+Rec_p_macro)
#Auc_p_macro = (Auc_p)
cat("Prediction phase-Acc=", Acc_p_macro, "\nPre=", Pre_p_macro, "\nRec=", Rec_p_macro, "\nF1=", F1_p_macro) # 输出计算的宏指标值

Acc_r_macro = (Acc_r)
Pre_r_macro = (Pre_r)
Rec_r_macro = (Rec_r)
F1_r_macro = 2*Pre_r_macro*Rec_r_macro/(Pre_r_macro+Rec_r_macro)
#Auc_r_macro = (Auc_r)
cat("Recall phase-Acc=", Acc_r_macro, "\nPre=", Pre_r_macro, "\nRec=", Rec_r_macro, "\nF1=", F1_r_macro)
