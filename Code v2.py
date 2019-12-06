import pandas as pd
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import scipy
from sklearn.model_selection import StratifiedKFold
import shap as sp
import lime as lm

def main():
    pd.options.mode.chained_assignment=None
    x,y=InputData()
    xgbm=xgb.XGBClassifier(scale_pos_weight=263/73,
                           learning_rate=0.007,
                           n_estimators=100,
                           gamma=0,
                           max_depth=4,
                           min_child_weight=2,
                           subsample=1,
                           eval_metric='error')
    rfm=RandomForestClassifier(n_estimators=100,
                               max_depth=4)
    lrm=LogisticRegression(solver='lbfgs')
    #ExperimentI(x,y,xgbm,rfm,lrm)
    #ExperimentII(x,y,xgbm,rfm,lrm)
    #ExperimentIII(x,y,rfm,lrm)
    #ExperimentIV(x,y,xgbm,rfm,lrm)
    #ExperimentV(x,y,xgbm,rfm,lrm)
    #ExperimentVI(x,y,xgbm,rfm,lrm)
    #ExperimentVII(xgbm,rfm,lrm)
    #ExperimentVIII(x,y,xgbm,rfm,lrm)

def InputData():
    dataset=pd.read_csv(os.getcwd()+'\\Dataset.csv')
    """
    print('Datasize: %d x %d'%(len(dataset.index),len(dataset.columns)))
    count=0
    for i in range(2,len(dataset.columns)):
        for j in range(0,len(dataset.index)):
            if pd.isna(dataset[dataset.columns[i]][j]):
                if count is 0:
                    print(dataset.columns[i],end=': ')
                count+=1
        if count is not 0:
            print(count)
            count=0
    """
    try:
        with tqdm(range(2,len(dataset.columns))) as bar:
            ftName=[]
            for i in bar:
                dataset[dataset.columns[i]]=dataset[dataset.columns[i]].fillna(dataset[dataset.columns[i]].mean())
                ftName.append(dataset.columns[i])
    except KeyboardInterrupt:
        bar.close()
        raise
    bar.close()
    x=dataset[ftName]
    y=dataset[dataset.columns[1]]
    return x,y

def ExperimentI(x,y,xgbm,rfm,lrm):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
    fig,ax=plt.subplots(1,1,figsize=(10,10))
    xgbm=xgbm.fit(x_train,y_train)
    y_pred=xgbm.predict_proba(x_test)[:,1]
    fpr,tpr,threshold=metrics.roc_curve(y_test,y_pred)
    roc_auc=metrics.auc(fpr,tpr)
    plt.plot(fpr,tpr,'r',label='XGBoost (AUC:%0.3F)'%roc_auc,linestyle='-')
    rfm=rfm.fit(x_train,y_train)
    y_pred=rfm.predict_proba(x_test)[:,1]
    fpr,tpr,threshold=metrics.roc_curve(y_test,y_pred)
    roc_auc=metrics.auc(fpr,tpr)
    plt.plot(fpr,tpr,'b',label='RandomForest (AUC:%0.3F)'%roc_auc,linestyle='-.')
    lrm=lrm.fit(x_train,y_train)
    y_pred=lrm.predict_proba(x_test)[:,1]
    fpr,tpr,threshold=metrics.roc_curve(y_test,y_pred)
    roc_auc=metrics.auc(fpr,tpr)
    plt.plot(fpr,tpr,'G',label='LogisticRegression (AUC:%0.3F)'%roc_auc,linestyle=':')
    plt.legend(loc='lower right',frameon=False)
    plt.show()
    
def ExperimentII(x,y,xgbm,rfm,lrm):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
    xgbm=xgbm.fit(x_train,y_train)
    y_pred=xgbm.predict_proba(x_test)[:,1]
    y_pred_XGB=np.array(y_pred)
    rfm=rfm.fit(x_train,y_train)
    y_pred=rfm.predict_proba(x_test)[:,1]
    y_pred_RF=np.array(y_pred)
    lrm=lrm.fit(x_train,y_train)
    y_pred=lrm.predict_proba(x_test)[:,1]
    y_pred_LR=np.array(y_pred)
    y_ans=np.array(y_test)
    print(delong_roc_test(y_ans,y_pred_XGB,y_pred_RF))
    print(delong_roc_test(y_ans,y_pred_XGB,y_pred_LR))
    
def ExperimentIII(x,y,rfm,lrm):
    kfold=StratifiedKFold(n_splits=5)
    xgbm=xgb.XGBClassifier(n_estimators=100,
                           max_depth=4,
                           eval_metric='error')
    AS=[[],[],[]]
    BS=[[],[],[]]
    for train,test in kfold.split(x,y):
        x_train=x.drop(index=test)
        x_test=x.drop(index=train)
        y_train=y.drop(index=test)
        y_test=y.drop(index=train)
        y_ans=np.array(y_test)
        xgbm=xgbm.fit(x_train,y_train)
        y_pred=xgbm.predict(x_test)
        y_prob=xgbm.predict_proba(x_test)[:,1]
        AS[0].append('%.4f'%metrics.accuracy_score(y_ans,y_pred))
        BS[0].append('%.4f'%metrics.brier_score_loss(y_ans,y_prob))
        rfm=rfm.fit(x_train,y_train)
        y_pred=rfm.predict(x_test)
        y_prob=rfm.predict_proba(x_test)[:,1]
        AS[1].append('%.4f'%metrics.accuracy_score(y_ans,y_pred))
        BS[1].append('%.4f'%metrics.brier_score_loss(y_ans,y_prob))
        lrm=lrm.fit(x_train,y_train)
        y_pred=lrm.predict(x_test)
        y_prob=lrm.predict_proba(x_test)[:,1]
        AS[2].append('%.4f'%metrics.accuracy_score(y_ans,y_pred))
        BS[2].append('%.4f'%metrics.brier_score_loss(y_ans,y_prob))
    for i in range(0,3):
        for j in range(0,5):
            AS[i][j]=float(AS[i][j])
            BS[i][j]=float(BS[i][j])
    for i in range(0,3):
        print('%.2f, %.2f'%(max(AS[i])*100,min(BS[i])*100))
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
    y_ans=np.array(y_test)
    xgbm=xgbm.fit(x_train,y_train)
    y_pred=xgbm.predict(x_test)
    print(metrics.precision_recall_fscore_support(y_ans,y_pred,average='macro'))
    rfm=rfm.fit(x_train,y_train)
    y_pred=rfm.predict(x_test)
    print(metrics.precision_recall_fscore_support(y_ans,y_pred,average='macro'))
    lrm=lrm.fit(x_train,y_train)
    y_pred=lrm.predict(x_test)
    print(metrics.precision_recall_fscore_support(y_ans,y_pred,average='macro'))

def ExperimentIV(x,y,xgbm,rfm,lrm):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
    fig,ax=plt.subplots(1,1,figsize=(10,10))
    xgbm=xgbm.fit(x_train,y_train)
    explainer=sp.TreeExplainer(xgbm)
    shap_values=explainer.shap_values(x_test)
    sp.summary_plot(shap_values,x_test)
    plt.show()
    sp.dependence_plot('PSI',shap_values,x_test,interaction_index='APACHEII')
    sp.dependence_plot('EarlyECMO',shap_values,x_test,interaction_index='ECMO')
    sp.dependence_plot('PSI',shap_values,x_test,interaction_index=None)
    sp.dependence_plot('APACHEII',shap_values,x_test,interaction_index=None)
    sp.dependence_plot('CumuD_4_balance',shap_values,x_test,interaction_index=None)
    
def ExperimentV(x,y,xgbm,rfm,lrm):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
    x_train=x_train.values
    x_test=x_test.values
    y_train=y_train.values
    y_test=y_test.values
    xgbm=xgbm.fit(x_train,y_train)
    explainer=lm.lime_tabular.LimeTabularExplainer(x_train,feature_names=x.columns,class_name=['Mortality_30days'])
    i=17
    exp=explainer.explain_instance(x_test[i],xgbm.predict_proba,num_features=5)
    exp.show_in_notebook(show_all=False)

def ExperimentVI(x,y,xgbm,rfm,lrm):
    df1=pd.read_csv(os.getcwd()+'\\Dataset.csv')
    df2=pd.DataFrame(columns=['No','Mortality_30days','XGB','RF','LR'])
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
    xgbm=xgbm.fit(x_train,y_train)
    y_pred_XGB=xgbm.predict_proba(x)
    rfm=rfm.fit(x_train,y_train)
    y_pred_RF=rfm.predict_proba(x)
    lrm=lrm.fit(x_train,y_train)
    y_pred_LR=lrm.predict_proba(x)
    try:
        with tqdm(range(0,len(df1))) as bar:
            for i in bar:
                temp=[]
                temp.append(df1['No'][i])
                temp.append(df1['Mortality_30days'][i])
                if temp[1]==0:
                    temp.append('%.4f'%y_pred_XGB[i][0])
                    temp.append('%.4f'%y_pred_RF[i][0])
                    temp.append('%.4f'%y_pred_LR[i][0])
                else:
                    temp.append('%.4f'%y_pred_XGB[i][1])
                    temp.append('%.4f'%y_pred_RF[i][1])
                    temp.append('%.4f'%y_pred_LR[i][1])
                df2.loc[i,df2.columns]=temp
    except KeyboardInterrupt:
        bar.close()
        raise
    bar.close()
    df2.to_csv(os.getcwd()+'\\Dataset for Hosmer-Lemeshow goodness of fit test.csv',index=False)

def ExperimentVII(xgbm,rfm,lrm):
    df=pd.read_excel(os.getcwd()+'\\Dataset for hospital.xlsx',sheet_name=0)
    df=df.rename(columns={df.columns[0]:'No',df.columns[1]:'Hospital'})
    dataset=pd.read_csv(os.getcwd()+'\\Dataset.csv')
    dataset=pd.merge(left=dataset,right=df,on='No')
    try:
        with tqdm(range(2,len(dataset.columns))) as bar:
            ftName=[]
            for i in bar:
                dataset[dataset.columns[i]]=dataset[dataset.columns[i]].fillna(dataset[dataset.columns[i]].mean())
                ftName.append(dataset.columns[i])
    except KeyboardInterrupt:
        bar.close()
        raise
    bar.close()
    x=dataset[ftName]
    y=dataset[dataset.columns[1]]
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
    fig,ax=plt.subplots(1,1,figsize=(10,10))
    xgbm=xgbm.fit(x_train,y_train)
    xgb.plot_importance(xgbm,max_num_features=50,ax=ax,importance_type='gain')
    plt.show()
    explainer=sp.TreeExplainer(xgbm)
    shap_values=explainer.shap_values(x_test)
    sp.summary_plot(shap_values,x_test)
    plt.show()
    sp.dependence_plot('Hospital',shap_values,x_test,interaction_index=None)
    plt.show()
    y_pred=xgbm.predict_proba(x_test)[:,1]
    fpr,tpr,threshold=metrics.roc_curve(y_test,y_pred)
    roc_auc=metrics.auc(fpr,tpr)
    plt.plot(fpr,tpr,'r',label='XGBoost (AUC:%0.3F)'%roc_auc,linestyle='-')
    rfm=rfm.fit(x_train,y_train)
    y_pred=rfm.predict_proba(x_test)[:,1]
    fpr,tpr,threshold=metrics.roc_curve(y_test,y_pred)
    roc_auc=metrics.auc(fpr,tpr)
    plt.plot(fpr,tpr,'b',label='RandomForest (AUC:%0.3F)'%roc_auc,linestyle='-.')
    lrm=lrm.fit(x_train,y_train)
    y_pred=lrm.predict_proba(x_test)[:,1]
    fpr,tpr,threshold=metrics.roc_curve(y_test,y_pred)
    roc_auc=metrics.auc(fpr,tpr)
    plt.plot(fpr,tpr,'G',label='LogisticRegression (AUC:%0.3F)'%roc_auc,linestyle=':')
    plt.legend(loc='lower right',frameon=False)
    plt.show()
    
def ExperimentVIII(x,y,xgbm,rfm,lrm):
    """
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
    fig,ax=plt.subplots(1,1,figsize=(10,10))
    xgbm=xgbm.fit(x_train,y_train)
    explainer=sp.TreeExplainer(xgbm)
    shap_values=explainer.shap_values(x_test)
    sp.summary_plot(shap_values,x_test,max_display=30)
    plt.show()
    rfm=rfm.fit(x_train,y_train)
    explainer=sp.TreeExplainer(rfm)
    shap_values=explainer.shap_values(x_test)
    sp.summary_plot(shap_values,x_test,max_display=30)
    plt.show()
    """
    dataset=pd.read_csv(os.getcwd()+'\\Dataset.csv')
    try:
        with tqdm(range(2,len(dataset.columns))) as bar:
            ftName=[]
            for i in bar:
                dataset[dataset.columns[i]]=dataset[dataset.columns[i]].fillna(dataset[dataset.columns[i]].mean())
                ftName.append(dataset.columns[i])
    except KeyboardInterrupt:
        bar.close()
        raise
    bar.close()
    x=dataset[['No','Mortality_30days','PSI','CumuD_4_balance','PaO2/FiO2','SteroidDay0_5_3groups','EarlyECMO',
        'APACHEII','D2Balance','D4input','BMI','Day0__PEEP',
        'Day0__Peak','D2input','ECMO','BUN','Na',
        'Cough','K','2nd bacterial complication','Vasopressor_3days','D3output',
        'D3input','Fever','Pltk','Hb','Day0__TV_PBW',
        'CRP','D2output','Age','Cr','Sex']]
    y=dataset[['No','Mortality_30days','PSI','APACHEII','D2output','CumuD_4_balance','CRP',
        'D2Balance','sBPst90','BUN','ECMO','Day0__TV_PBW',
        'K','Cr','SteroidDay0_5_3groups','HR','BMI',
        'Na','D3input','EarlyECMO','D2input','Age',
        'Day0__FiO2','PaO2/FiO2','WBC','Day0__PEEP','FreshHD',
        'Pltk','Hb','D4input','1st dose Tamiflu','Flu_A']]
    x.to_csv(os.getcwd()+'//Dataset for HL-test (XGB Top-30).csv',index=False)
    y.to_csv(os.getcwd()+'//Dataset for HL-test (RF Top-30).csv',index=False)

# --------------------------------------------------
# Source: https://github.com/yandexdataschool/roc_comparison/blob/master/compare_auc_delong_xu.py
def compute_midrank(x):
    J=np.argsort(x)
    Z=x[J]
    N=len(x)
    T=np.zeros(N,dtype=np.float)
    i=0
    while i<N:
        j=i
        while j<N and Z[j]==Z[i]:
            j+=1
        T[i:j]=0.5*(i+j-1)
        i=j
    T2=np.empty(N,dtype=np.float)
    T2[J]=T+1
    return T2

def fastDeLong(predictions_sorted_transposed,label_1_count):
    m=label_1_count
    n=predictions_sorted_transposed.shape[1]-m
    positive_examples=predictions_sorted_transposed[:,:m]
    negative_examples=predictions_sorted_transposed[:,m:]
    k=predictions_sorted_transposed.shape[0]
    tx=np.empty([k,m],dtype=np.float)
    ty=np.empty([k,n],dtype=np.float)
    tz=np.empty([k,m+n],dtype=np.float)
    for r in range(k):
        tx[r,:]=compute_midrank(positive_examples[r,:])
        ty[r,:]=compute_midrank(negative_examples[r,:])
        tz[r,:]=compute_midrank(predictions_sorted_transposed[r,:])
    aucs=tz[:,:m].sum(axis=1)/m/n-float(m+1.0)/2.0/n
    v01=(tz[:,:m]-tx[:,:])/n
    v10=1.0-(tz[:,m:]-ty[:,:])/m
    sx=np.cov(v01)
    sy=np.cov(v10)
    delongcov=sx/m+sy/n
    return aucs,delongcov

def calc_pvalue(aucs,sigma):
    l=np.array([[1,-1]])
    z=np.abs(np.diff(aucs)/np.sqrt(np.dot(np.dot(1,sigma),l.T)))
    return np.log10(2)+scipy.stats.norm.logsf(z,loc=0,scale=1)/np.log(10)

def compute_ground_truth_statistics(ground_truth,sample_weight=None):
    assert np.array_equal(np.unique(ground_truth),[0,1])
    order=(-ground_truth).argsort()
    label_1_count=int(ground_truth.sum())
    if sample_weight is None:
        ordered_sample_weight=None
    else:
        ordered_sample_weight=sample_weight[order]
    return order,label_1_count,ordered_sample_weight

def delong_roc_test(ground_truth,predictions_one,predictions_two):
    sample_weight=None
    order,label_1_count,ordered_sample_weight=compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed=np.vstack((predictions_one,predictions_two))[:,order]
    aucs,delongcov=fastDeLong(predictions_sorted_transposed,label_1_count)
    return calc_pvalue(aucs,delongcov)
# --------------------------------------------------

if __name__ is '__main__':
    main()