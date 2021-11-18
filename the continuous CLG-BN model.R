# Program for learning continuous CLGBN for liquefaction identification
library(bnlearn) # Load related dependent software packages
library(Rgraphviz) 
library(gmodels) 
library(pROC) 

compare = function(x) { # Custom function, used to classify the continuous liquefaction discriminant node 
	if (x > 0.5) {
		x = 1
	} else {
		x = 0
	}
}

# Read data 
data = read.table("Vs_Continuous.txt", header = T, sep = ",") 
data$Liq = as.numeric(data$Liq) 
#Random sampling 
n=nrow(data)
ceiling(0.9*660)
trainindex <- sample(1:n, 594)
Training_data <- data[trainindex,]
head(Training_data)
Test_data <- data[-trainindex,]
#data$PGA = log10(Con_data$PGA) 
#data$Vs = log10(Con_data$Vs)

## Structure learning 
#bn_hc = hc(Con_data, whitelist = NULL, blacklist = NULL) #Use hill climbing algorithm to learn the initial network structure 
#graphviz.plot(bn_hc, shape = "rectangle") #
## Revise structure with expert knowledge # 
#bn_liq = drop.arc(bn_hc, from = "Mw", to = "Dw") # 
#bn_liq = reverse.arc(bn_liq, from = "Vs1", to = "Sv0", check.cycles = T, check.illegal = T) # 
#bn_liq = set.arc(bn_liq, from = "Ts", to = "Liq", check.cycles = T, check.illegal = T) # 
#graphviz.plot(bn_liq, shape = "rectangle") # 

# Take the network structure obtained by structural learning as a fixed input 
bn_liq = model2network("[Mw][Ds][Ts][Dw][amax|Mw:Dw][Sv0|Ds:Dw][Vs1|Sv0][Liq|Mw:amax:Vs1:Ts]") 
graphviz.plot(bn_liq, shape = "rectangle") # Visualize the network structure

Acc_p = c() # 
Acc_r = c() # 
Pre_p = c() # 
Pre_r = c() # 
Rec_p = c() # 
Rec_r = c() # 
F1_p = c() # 
F1_r = c() # 
Auc_p = c() #
Auc_r = c() # 

	# Parameter learning 
	bn_parameter <- bn.fit(bn_liq, Training_data) 

	# Predict 
	pred_prob = predict(object = bn_parameter, node = "Liq", data = Test_data, method = "bayes-lw") 
	pred_bn = sapply(pred_prob, compare) 
	outcome = as.data.frame(table(pred_bn, Test_data[, "Liq"])) 
	Acc_p[i] = (outcome[1,3]+outcome[4,3])/nrow(Test_data) 
	Pre_p[i] = outcome[4,3]/(outcome[4,3]+outcome[2,3])
	Rec_p[i] = outcome[4,3]/(outcome[4,3]+outcome[3,3])
	Roc_p = roc(Test_data[, "Liq"], pred_prob) 
	Auc_p[i] = auc(Roc_p) 
 	CrossTable(pred_bn, Test_data$Liq, prop.chisq=F, prop.r=F, prop.c=F, prop.t=T) 

 	# Recall 
	pred_prob = predict(object = bn_parameter, node = "Liq", data = Training_data, method = "bayes-lw") 
	pred_bn = sapply(pred_prob, compare)
	outcome = as.data.frame(table(pred_bn, Training_data[, "Liq"]))
	Acc_r[i] = (outcome[1,3]+outcome[4,3])/nrow(Training_data)
	Pre_r[i] = outcome[4,3]/(outcome[4,3]+outcome[2,3])
	Rec_r[i] = outcome[4,3]/(outcome[4,3]+outcome[3,3])
	Roc_r = roc(Training_data[, "Liq"], pred_prob)
	Auc_r[i] = auc(Roc_r)
 	CrossTable(pred_bn, Training_data$Liq, prop.chisq=F, prop.r=F, prop.c=F, prop.t=T)
}

# Macro indicators for the forecast section 
Acc_p_macro =(Acc_p) 
Pre_p_macro = (Pre_p)
Rec_p_macro = (Rec_p)
F1_p_macro = 2*Pre_p_macro*Rec_p_macro/(Pre_p_macro+Rec_p_macro)
Auc_p_macro = (Auc_p)
cat("Prediction phase-Acc=", Acc_p_macro, "\nPre=", Pre_p_macro, "\nRec=", Rec_p_macro, "\nF1=", F1_p_macro, "\nAuc=", Auc_p_macro) # 输出宏指标值

# Macro indicators of the return part 
Acc_r_macro = Acc_r)
Pre_r_macro = (Pre_r)
Rec_r_macro = (Rec_r)
F1_r_macro = 2*Pre_r_macro*Rec_r_macro/(Pre_r_macro+Rec_r_macro)
Auc_r_macro = (Auc_r)
cat("Recall phase-Acc=", Acc_r_macro, "\nPre=", Pre_r_macro, "\nRec=", Rec_r_macro, "\nF1=", F1_r_macro, "\nAuc=", Auc_r_macro)