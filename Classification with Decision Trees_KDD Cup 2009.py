# 0.IMPORT 
# 0.1. PACKAGES

#seaborn
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pingouin as pg
import category_encoders as ce # Binary Encoding
import xgboost as xgb # XGBoost

# sklearn
from sklearn.model_selection import train_test_split # Train Test Split
from sklearn.impute import SimpleImputer # Imputation
from sklearn.preprocessing import LabelEncoder # LabelEncoder
from sklearn.naive_bayes import CategoricalNB # Naive Bayes
from sklearn.naive_bayes import MultinomialNB # Naive Bayes
from sklearn.linear_model import LogisticRegression # Logistic Regression
from sklearn.tree import DecisionTreeClassifier  # Decision Tree Classifier
from sklearn.ensemble import RandomForestClassifier  # Random Forest Classifier
from sklearn.ensemble import AdaBoostClassifier  # AdaBoostClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier  # LightGBM
from sklearn.utils import resample  # Upsampling
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import confusion_matrix  # Confusion Matrix
from sklearn import metrics  # ROC Curve
from sklearn.metrics import roc_auc_score  # ROC Score
from sklearn.metrics import plot_roc_curve as plt_roc
from sklearn.model_selection import GridSearchCV  # Hyperparameter Tuning
from sklearn.model_selection import ShuffleSplit

# scipy
from scipy import stats
from scipy.stats import sem

#statsmodels
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif # Variance Inflation Factor
from statsmodels.tools.tools import add_constant # Add Constant
import statsmodels.discrete.discrete_model as sm1

# 0.2. Import Data
test = pd.read_csv(
    '', sep='\t')
train = pd.read_csv(
    '', sep='\t')
churn = pd.read_csv(
    'http://www.vincentlemaire-labs.fr/kddcup2009/orange_small_train_churn.labels', header=None)
appetency = pd.read_csv(
    'https://www.kdd.org/cupfiles/KDDCupData/2009/orange_small_train_appetency.labels', header=None)
upselling = pd.read_csv(
    'https://www.kdd.org/cupfiles/KDDCupData/2009/orange_small_train_upselling.labels', header=None)

# Transform dependent variables: (-1,1) -> (0,1)
churn = np.where(churn[0] == 1, 1, 0)
appetency = np.where(appetency[0] == 1, 1, 0)
upselling = np.where(upselling[0] == 1, 1, 0)

# LOOP (to create mean of 10 AUCs)
# Currently deselected. Only apply to calculate final AUC Score (mean of 10 AUCs).
# for i in range(10):
# rnd = i
# 80/20 split to train & validation for each variable

###
# 0.2.  HELPER METHODS
# 0.2.1. DETECT OUTLIERS
outliers = []
def detect_outlier(data_1):
    threshold = 4
    mean_1 = np.mean(data_1)
    std_1 = np.std(data_1)

    for y in data_1:
        z_score = (y - mean_1)/std_1
        if np.abs(z_score) > threshold:
            outliers.append(y)
    return outliers
###

# 1.PREPROCESSING
# 1.1. STRATIFIED SAMPLE
# Known issue of imbalanced data set.
# 'Consider testing random and non-random (e.g. stratified) sampling schemes.'

# Define random state
rnd = 10

# Train/Test Split (80/20)
X_train_chu, X_val_chu, chu_train, chu_val = train_test_split(
    train, churn, test_size=0.2, random_state=rnd, stratify=churn)
X_train_app, X_val_app, app_train, app_val = train_test_split(
    train, appetency, test_size=0.2, random_state=rnd,stratify = appetency)
X_train_up, X_val_up, up_train, up_val = train_test_split(
    train, upselling, test_size=0.2, random_state=rnd, stratify=upselling)

# Transform dependent variables: Series -> Dataframe
chu_train = pd.DataFrame(chu_train)
chu_val = pd.DataFrame(chu_val)
app_train = pd.DataFrame(app_train)
app_val = pd.DataFrame(app_val)
up_train = pd.DataFrame(up_train)
up_val = pd.DataFrame(up_val)

# 1.1. IMPUTATION OF MISSING VALUES
# Source 4: https://towardsdatascience.com/how-to-handle-missing-data-8646b18db0d4
# Source 5: https://towardsdatascience.com/handling-missing-data-for-a-beginner-6d6f5ea53436
# Source 6: https://towardsdatascience.com/6-different-ways-to-compensate-for-missing-values-data-imputation-with-examples-6022d9ca0779

# Are there columns that contain no information?
train_isna = X_train_chu.isna().sum().to_frame()
train_0 = train_isna[train_isna[0]==50000]

# What percentage of values are missing in all other columns.
train_isna = X_train_chu.isna().sum().to_frame()
train_isna_per = (train_isna / 40000)*100

# Visualisation 1: Histogram
train_isna_per = train_isna_per.sort_values(by=[0])
plt.hist(train_isna_per[0], bins = 10)
plt.show()
# Visualisation 2: Scatterplot
plt.title('Percentage of Missing Values by Variable')
plt.ylabel("percentage")
plt.xlabel("variable")
plt.tick_params(axis='x', bottom=False, labelbottom=False) 
plt.scatter(train_isna_per.index, train_isna_per.values, s=0.8, marker=',')
plt.show()
# Visualisation 3: Line Plot how much is missing max
miss_per = train.isna().sum()/50000
miss_per_cont = train.iloc[:,:190].isna().sum()/50000
miss_per_cat = train.iloc[:, 190:].isna().sum()/50000
miss = pd.DataFrame(columns=['Full','Cont','Cat'],index = range(101))
for i in range(101):
    if i == 0:
        miss.iloc[i, 0] = (len( miss_per[miss_per > ( i/100) ] ) /230) *100
        miss.iloc[i, 1] = (len(miss_per_cont[miss_per_cont > (i/100)])/190)*100
        miss.iloc[i, 2] = (len(miss_per_cat[miss_per_cat > (i/100)])/40)*100
    else:
        miss.iloc[i, 0] = (len(miss_per[miss_per >= (i/100)])/230)*100
        miss.iloc[i, 1] = (len(miss_per_cont[miss_per_cont >= (i/100)])/190)*100
        miss.iloc[i, 2] = (len(miss_per_cat[miss_per_cat >= (i/100)])/40)*100

f=plt.figure()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(miss['Full'], label=r'\textbf{All}', color='seagreen')
plt.plot(miss['Cont'], label=r'\textbf{Continous}', color ='rebeccapurple')
plt.plot(miss['Cat'], label=r'\textbf{Categorical}', color ='indianred')
plt.legend(loc='upper right', prop={'size': 8})
plt.xlabel(r"\textbf{Missing Values \%}")
plt.ylabel(r"\textbf{Variables \%}")
f.savefig("MissingValues.pdf", bbox_inches='tight')
plt.show()

# We can not be sure which category our missing data belongs to. Hence we choose to impute all missing values.

# Drop all variables without information
for variable in X_train_chu.columns:
    if X_train_chu[variable].nunique() < 2:
       X_train_chu = X_train_chu.drop([variable], axis=1)
       X_val_chu = X_val_chu.drop([variable], axis=1)
    if X_train_app[variable].nunique() < 2:
       X_train_app = X_train_app.drop([variable], axis=1)
       X_val_app = X_val_app.drop([variable], axis=1)
    if X_train_up[variable].nunique() < 2:
       X_train_up = X_train_up.drop([variable], axis=1)
       X_val_up = X_val_up.drop([variable], axis=1)

# Delete all columns with >90% missing values
for variable in X_train_chu.columns:
    if X_train_chu[variable].isna().sum()/len(X_train_chu[variable]) > 0.9:
       X_train_chu = X_train_chu.drop([variable], axis=1)
       X_val_chu = X_val_chu.drop([variable], axis=1)
for variable in X_train_app.columns:
    if X_train_app[variable].isna().sum()/len(X_train_app[variable]) > 0.9:
       X_train_app = X_train_app.drop([variable], axis=1)
       X_val_app = X_val_app.drop([variable], axis=1)
for variable in X_train_up.columns:
    if X_train_up[variable].isna().sum()/len(X_train_up[variable]) > 0.9:
       X_train_up = X_train_up.drop([variable], axis=1)
       X_val_up = X_val_up.drop([variable], axis=1)

# 1.1.1 CATEGORICAL VARIABLES
# Replace missing values with fixed value (seperate category)
imp_const = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='Missing')
imp_const.fit(X_train_chu.loc[:,'Var192':])
X_train_chu.loc[:,'Var192':] = imp_const.transform(X_train_chu.loc[:,'Var192':])
X_val_chu.loc[:,'Var192':] = imp_const.transform(X_val_chu.loc[:,'Var192':])

imp_const.fit(X_train_app.loc[:, 'Var192':])
X_train_app.loc[:, 'Var192':] = imp_const.transform(X_train_app.loc[:, 'Var192':])
X_val_app.loc[:, 'Var192':] = imp_const.transform(X_val_app.loc[:, 'Var192':])

imp_const.fit(X_train_up.loc[:, 'Var192':])
X_train_up.loc[:, 'Var192':] = imp_const.transform(X_train_up.loc[:, 'Var192':])
X_val_up.loc[:, 'Var192':] = imp_const.transform(X_val_up.loc[:, 'Var192':])

# ALTERNATIVE
# # Source 4 recommends to replace missing values with new values only for variables with few missing values.
# # Create new level for all categorical variables with only few missing values.
# # <= 1.5% missing data: create new characteristic
# X_train.fillna({'Var192':'Empty', 'Var197':'Empty','Var199':'Empty','Var202':'Empty','Var203':'Empty','Var208':'Empty','Var217':'Empty','Var218':'Empty'}, inplace=True)
# X_val.fillna({'Var192':'Empty', 'Var197':'Empty','Var199':'Empty','Var202':'Empty','Var203':'Empty','Var208':'Empty','Var217':'Empty','Var218':'Empty'}, inplace=True)

# # > 1.5% missing data:
# # If < 3 characteristics: create new characteristic
# X_train.fillna({'Var191':'Empty', 'Var201':'Empty','Var213':'Empty','Var215':'Empty','Var224':'Empty'}, inplace=True)
# X_val.fillna({'Var191':'Empty', 'Var201':'Empty','Var213':'Empty','Var215':'Empty','Var224':'Empty'}, inplace=True)

# # else: replace with most frequent
# imp_most = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
# imp_most.fit(X_train.loc[:,'Var191':])
# X_train.loc[:,'Var191':] = imp_most.transform(X_train.loc[:,'Var191':])
# X_val.loc[:,'Var191':] = imp_most.transform(X_val.loc[:,'Var191':])

# 1.1.2. CONTINOUS VARIABLES
# Boxplots of Continous Variables
feat_cont = train.iloc[:, :190]
sns.boxplot(data=feat_cont)
plt.show()
# Impuations with Mean
var1 = 'Var189'
X_train_chu.loc[:,:var1]=X_train_chu.loc[:,:var1].fillna(X_train_chu.loc[:,:var1].mean())
X_val_chu.loc[:,:var1]=X_val_chu.loc[:,:var1].fillna(X_train_chu.loc[:,:var1].mean())

X_train_app.loc[:, :var1] = X_train_app.loc[:, :var1].fillna(X_train_app.loc[:, :var1].mean())
X_val_app.loc[:, :var1] = X_val_app.loc[:, :var1].fillna(X_train_app.loc[:, :var1].mean())

X_train_up.loc[:, :var1] = X_train_up.loc[:, :var1].fillna(X_train_up.loc[:, :var1].mean())
X_val_up.loc[:, :var1] = X_val_up.loc[:, :var1].fillna(X_train_up.loc[:, :var1].mean())

# 1.2. CATERGORICAL VARIABLES WITH MANY FEATURES
# Reset Indices
X_train_chu = X_train_chu.reset_index()
X_train_chu = X_train_chu.drop(['index'],axis=1)
X_train_app = X_train_app.reset_index()
X_train_app = X_train_app.drop(['index'], axis=1)
X_train_up = X_train_up.reset_index()
X_train_up = X_train_up.drop(['index'], axis=1)
X_val_chu = X_val_chu.reset_index()
X_val_chu = X_val_chu.drop(['index'],axis=1)
X_val_app = X_val_app.reset_index()
X_val_app = X_val_app.drop(['index'], axis=1)
X_val_up = X_val_up.reset_index()
X_val_up = X_val_up.drop(['index'], axis=1)

# Show range of individual features in categorical variables
feat_cat = train.iloc[:, 190:]
y = pd.DataFrame()
x = pd.DataFrame()
for variable in feat_cat.columns:
    x = pd.DataFrame(feat_cat[variable].value_counts()/feat_cat[variable].notna().sum())
    x = x.reset_index()
    y = pd.concat([y, x.iloc[:,1]], ignore_index=True, axis=1)
y.columns = feat_cat.columns
v = plt.figure()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
sns.catplot(data=y, kind="strip")
plt.xticks(rotation=90, fontsize=7)
plt.xlabel(r"\textbf{Categorical Variable}")
plt.ylabel(r"\textbf{Proportion of Feature Occurence}")
plt.savefig("CategoricalValues.pdf", bbox_inches='tight')
plt.show()
    
# Several Variables have many features with only little frequency

# 1.2.1 RARE LABEL ENCODING
# Source 8: https: // heartbeat.fritz.ai/hands-on-with-feature-engineering-techniques-encoding-categorical-variables-be4bc0715394
# 'Rare labels may cause some issues, especially with overfitting and generalization. '
threshlold = 0.0005
# we loop over all the categorical variables
num1 = 41
for variable in X_train_chu.columns:
    # Prior 189
    if X_train_chu.columns.get_loc(variable) > num1:
        # locate all the categories that are not rare.
        counts = X_train_chu.groupby([variable])[variable].count() / len(X_train_chu)
        frequent_labels = [x for x in counts.loc[counts > threshlold].index.values]

        # change the rare category names with the word rare, and thus encoding it.
        X_train_chu[variable] = np.where(X_train_chu[variable].isin(frequent_labels), X_train_chu[variable], 'Rare')
        X_val_chu[variable] = np.where(X_val_chu[variable].isin(frequent_labels), X_val_chu[variable], 'Rare')

for variable in X_train_app.columns:
    if X_train_app.columns.get_loc(variable) > num1:
        counts = X_train_app.groupby([variable])[variable].count() / len(X_train_app)
        frequent_labels = [x for x in counts.loc[counts > threshlold].index.values]

        X_train_app[variable] = np.where(X_train_app[variable].isin(frequent_labels), X_train_app[variable], 'Rare')
        X_val_app[variable] = np.where(X_val_app[variable].isin(frequent_labels), X_val_app[variable], 'Rare')

for variable in X_train_up.columns:
    if X_train_up.columns.get_loc(variable) > num1:
        counts = X_train_up.groupby([variable])[variable].count() / len(X_train_up)
        frequent_labels = [x for x in counts.loc[counts > threshlold].index.values]

        X_train_up[variable] = np.where(X_train_up[variable].isin(frequent_labels), X_train_up[variable], 'Rare')
        X_val_up[variable] = np.where(X_val_up[variable].isin(frequent_labels), X_val_up[variable], 'Rare')

# ALTERNATIVE
# # Frequency encoding categorical variables with >=1000 factors
# # (https://www.kaggle.com/bhavikapanara/frequency-encoding)
# group = [198, 199, 200, 202, 214, 216, 217, 220, 222]
# for i in group:
#     enc = (X_train.groupby('Var{}'.format(i)).size())
#     for index, row in X_val.iterrows():
#         if X_train['Var{}'.format(i)].str.contains(X_val.iloc[index, X_val.columns.get_loc('Var{}'.format(i))]).any():
#             X_val.iloc[index, X_val.columns.get_loc('Var{}'.format(
#                 i))] = enc[X_val.iloc[index, X_val.columns.get_loc('Var{}'.format(i))]]
#         else:
#             X_val.iloc[index, X_val.columns.get_loc('Var{}'.format(i))] = 1
#     X_train['Var{}'.format(i)] = X_train['Var{}'.format(i)
#                                          ].apply(lambda x: enc[x])

# full_train = X_train.append(X_val)

# DROP VARIABLES (without information)
for variable in X_train_chu.columns:
    if X_train_chu[variable].nunique() < 2:
       X_train_chu = X_train_chu.drop([variable], axis=1)
       X_val_chu = X_val_chu.drop([variable], axis=1)

for variable in X_train_app.columns:
    if X_train_app[variable].nunique() < 2:
       X_train_app = X_train_app.drop([variable], axis=1)
       X_val_app = X_val_app.drop([variable], axis=1)

for variable in X_train_up.columns:
    if X_train_up[variable].nunique() < 2:
       X_train_up = X_train_up.drop([variable], axis=1)
       X_val_up = X_val_up.drop([variable], axis=1)

# 1.4. OUTLIERS (in continous variables)
# 1.4.1. EXPLORATION
# # Do the continous variables have outliers?
# outlier_n = pd.DataFrame(columns=['Variable','Outliers'])
# for variable in X_train.columns:
#     outliers = []
#     if X_train.columns.get_loc(variable) < 172:
#        outlier_n = outlier_n.append(
#            {'Variable': variable, 'Outliers': len(detect_outlier(X_train[variable]))}, ignore_index=True)

# 1.5. HANDLE CATEGORICAL VARIABLES
# 1.5.1. ONE HOT ENCODING 
# Label Encoding (categorical variables)
le = LabelEncoder()
for variable in X_train_chu.columns:
    if X_train_chu.columns.get_loc(variable) > num1: # XXXXXX  Check correct number
        le.fit(X_train_chu[variable])
        X_train_chu[variable] = le.transform(X_train_chu[variable])
        X_val_chu[variable] = le.transform(X_val_chu[variable])

for variable in X_train_app.columns:
    if X_train_app.columns.get_loc(variable) > num1:  # XXXXXX  Check correct number
        le.fit(X_train_app[variable])
        X_train_app[variable] = le.transform(X_train_app[variable])
        X_val_app[variable] = le.transform(X_val_app[variable])

for variable in X_train_up.columns:
    if X_train_up.columns.get_loc(variable) > num1:  # XXXXXX  Check correct number
        le.fit(X_train_up[variable])
        X_train_up[variable] = le.transform(X_train_up[variable])
        X_val_up[variable] = le.transform(X_val_up[variable])


# One Hot Encoding
X_full_chu = pd.concat([X_train_chu, X_val_chu])
X_full_chu = pd.get_dummies(data=X_full_chu, columns=['Var192', 'Var193', 'Var194','Var195', 'Var196', 'Var197', 'Var198', 'Var199', 'Var200', 'Var201','Var202', 'Var203', 'Var204', 'Var205', 'Var206','Var207', 'Var208', 'Var210', 'Var211', 'Var212', 'Var214', 'Var216', 'Var217', 'Var218', 'Var219', 'Var220', 'Var221', 'Var222', 'Var223', 'Var225', 'Var226', 'Var227', 'Var228', 'Var229'])
X_train_chu = X_full_chu.iloc[:40000, :]
X_val_chu = X_full_chu.iloc[40000:, :]

X_full_app = pd.concat([X_train_app, X_val_app])
X_full_app = pd.get_dummies(data=X_full_app, columns=['Var192', 'Var193', 'Var194', 'Var195', 'Var196', 'Var197', 'Var198', 'Var199', 'Var200', 'Var201', 'Var202', 'Var203', 'Var204', 'Var205', 'Var206','Var207', 'Var208', 'Var210', 'Var211', 'Var212', 'Var214', 'Var216', 'Var217', 'Var218', 'Var219', 'Var220', 'Var221', 'Var222', 'Var223', 'Var225', 'Var226', 'Var227', 'Var228', 'Var229'])
X_train_app = X_full_app.iloc[:40000, :]
X_val_app = X_full_app.iloc[40000:, :]

X_full_up = pd.concat([X_train_up, X_val_up])
X_full_up = pd.get_dummies(data=X_full_up, columns=['Var192', 'Var193', 'Var194', 'Var195', 'Var196', 'Var197', 'Var198', 'Var199', 'Var200', 'Var201', 'Var202', 'Var203', 'Var204', 'Var205', 'Var206','Var207', 'Var208', 'Var210', 'Var211', 'Var212', 'Var214', 'Var216', 'Var217', 'Var218', 'Var219', 'Var220', 'Var221', 'Var222', 'Var223', 'Var225', 'Var226', 'Var227', 'Var228', 'Var229'])
X_train_up = X_full_up.iloc[:40000, :]
X_val_up = X_full_up.iloc[40000:, :]

# Alternative I
# BINARY ENCODING (categorical variables)
# Source 10: https://medium.com/data-design/visiting-categorical-features-and-encoding-in-decision-trees-53400fa65931
# Performs better with decision trees than one hot encoding.
# X_train_o = X_train.copy()
# for variable in X_train_o.columns:
#     # Prior 189
#     if X_train.columns.get_loc(variable) > 41:
#         encoder = ce.BinaryEncoder(cols=[variable])
#         X_train = encoder.fit_transform(X_train)
#         X_val = encoder.fit_transform(X_val)

# 1.6. INBALANCE (in dependent variables)
# How balanced are the dependent variables?
chu_balance = len(chu_train[chu_train[0]==1])*100/len(X_train_chu)
app_balance = len(app_train[app_train[0] == 1])*100/len(X_train_chu)
up_balance = len(up_train[up_train[0] == 1])*100/len(X_train_chu)

# 1.6.1. UP-SAMPLE (minority class)
# Source 11: https://elitedatascience.com/imbalanced-classes
# Add outcome variables to train set
X_train_chu['churn'] = chu_train
X_train_app['appetency'] = app_train
X_train_up['upselling'] = up_train

# Separate majority and minority classes
X_majority_chu = X_train_chu[X_train_chu['churn'] == 0]
X_minority_chu = X_train_chu[X_train_chu['churn'] == 1]
X_majority_app = X_train_app[X_train_app['appetency'] == 0]
X_minority_app = X_train_app[X_train_app['appetency'] == 1]
X_majority_up = X_train_up[X_train_up['upselling'] == 0]
X_minority_up = X_train_up[X_train_up['upselling'] == 1]

# Upsample minority class
X_minority_chu_upsampled = resample(X_minority_chu,replace=True,n_samples=int(len(X_majority_chu)),random_state=rnd)
X_minority_app_upsampled = resample(X_minority_app, replace=True, n_samples=int(len(X_majority_app)), random_state=rnd)
X_minority_up_upsampled = resample(X_minority_up, replace=True, n_samples=int(len(X_majority_up)), random_state=rnd)

# Combine majority class with upsampled minority class
X_chu_upsampled = pd.concat([X_majority_chu, X_minority_chu_upsampled])
X_app_upsampled = pd.concat([X_majority_app, X_minority_app_upsampled])
X_up_upsampled = pd.concat([X_majority_up, X_minority_up_upsampled])

# Split into predictors and outcome variables
X_train_chu = X_chu_upsampled.drop(['churn'], axis=1)
chu_train = X_chu_upsampled.churn
X_train_app = X_app_upsampled.drop(['appetency'], axis=1)
app_train = X_app_upsampled.appetency
X_train_up = X_up_upsampled.drop(['upselling'], axis=1)
up_train = X_up_upsampled.upselling

# ALTERNATIVE I
# DOWN-SAMPLE (majority class)
# Separate majority and minority classes
# X_majority_app = X_train[X_train.appetency == 0]
# X_minority_app = X_train[X_train.appetency == 1]

# # Downsample majority class
# df_majority_downsampled = resample(X_majority_app,
#                                    replace=False,    # sample without replacement
#                                    n_samples=len(X_minority_app),     # to match minority class
#                                    random_state=5)  # reproducible results

# # Combine minority class with downsampled majority class
# df_downsampled = pd.concat([df_majority_downsampled, X_minority_app])

# X_train_app = df_downsampled.drop(['churn', 'appetency', 'upselling'], axis=1)
# app_train = df_downsampled.appetency

#%% 2. ANALYSIS
#%% 2.1. Decision Tree Classifier
DTC = DecisionTreeClassifier(class_weight='balanced')
DTC_chu = DTC.fit(X_train_chu,chu_train)
DTC_chu_pred = pd.DataFrame(DTC_chu.predict_proba(X_val_chu))
DTC_app = DTC.fit(X_train_app, app_train)
DTC_app_pred = pd.DataFrame(DTC_app.predict_proba(X_val_app))
DTC_up = DTC.fit(X_train_up, up_train)
DTC_up_pred = pd.DataFrame(DTC_up.predict_proba(X_val_up))

# Ensemble Methods
#%% 2.2. Random Tree Classifier
RTF = RandomForestClassifier(class_weight='balanced')
RTF_chu = RTF.fit(X_train_chu, chu_train)
RTF_chu_pred = pd.DataFrame(RTF_chu.predict_proba(X_val_chu))
RTF_app = RTF.fit(X_train_app, app_train)
RTF_app_pred = pd.DataFrame(RTF_app.predict_proba(X_val_app))
RTF_up = RTF.fit(X_train_up, up_train)
RTF_up_pred = pd.DataFrame(RTF_up.predict_proba(X_val_up))

#%% 2.3. Adaptive Boosting Classifier
AB = AdaBoostClassifier()
AB_chu = AB.fit(X_train_chu, chu_train)
AB_chu_pred = pd.DataFrame(AB_chu.predict_proba(X_val_chu))
AB_app = AB.fit(X_train_app, app_train)
AB_app_pred = pd.DataFrame(AB_app.predict_proba(X_val_app))
AB_up = AB.fit(X_train_up, up_train)
AB_up_pred = pd.DataFrame(AB_up.predict_proba(X_val_up))

#%% 2.4. Histogram-based Gradient Boosting Classification Tree (Light GBM)

# Hyperparameter Tuning using GridSearchCV
# tuning = GridSearchCV(estimator=HistGradientBoostingClassifier(), param_grid=p,scoring='roc_auc', cv=ShuffleSplit(test_size=0.20, n_splits=1, random_state=0))
# Solution
# p = {'learning_rate': [0.05],'max_leaf_nodes': [15],'max_iter': [80]}

# Implementation
LGBM = HistGradientBoostingClassifier(learning_rate=0.05, max_leaf_nodes=15, max_iter=80)
# Churn
LGBM_chu = LGBM.fit(X_train_chu, chu_train)
LGBM_chu_pred = pd.DataFrame(LGBM_chu.predict(X_val_chu))
LGBM_chu_pred_proba = pd.DataFrame(LGBM_chu.predict_proba(X_val_chu))
#fig, axi = plt.subplots()
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')
#plt_roc(LGBM_chu, X_val_chu, chu_val, name='Churn',ax=axi)
# Appetency
LGBM_app = LGBM.fit(X_train_app, app_train)
LGBM_app_pred = pd.DataFrame(LGBM_app.predict(X_val_app))
LGBM_app_pred_proba = pd.DataFrame(LGBM_app.predict_proba(X_val_app))
#plt_roc(LGBM_app, X_val_app, app_val, name='Appetency',ax=axi)
# Up-selling
LGBM_up = LGBM.fit(X_train_up, up_train)
LGBM_up_pred = pd.DataFrame(LGBM_up.predict(X_val_up))
LGBM_up_pred_proba = pd.DataFrame(LGBM_up.predict_proba(X_val_up))
#plt_roc(LGBM_up, X_val_up, up_val, name='Up-selling',ax=axi)
#plt.savefig("ROC.pdf", bbox_inches='tight')
#plt.show()
#%% 2.5. XGB Classifier
XGB = xgb.XGBClassifier()
XGB_chu = XGB.fit(X_train_chu, chu_train)
XGB_chu_pred = pd.DataFrame(XGB_chu.predict_proba(X_val_chu))
XGB_app = XGB.fit(X_train_app, app_train)
XGB_app_pred = pd.DataFrame(XGB_app.predict_proba(X_val_app))
XGB_up = XGB.fit(X_train_up, up_train) 
XGB_up_pred = pd.DataFrame(XGB_up.predict_proba(X_val_up))

## 2.6 Sum up AUC_scores
# a += roc_auc(chu_val,LGBM_chu_pred_proba[1])
# b += roc_auc(app_val,LGBM_app_pred_proba[1])
# c += roc_auc(up_val,LGBM_up_pred_proba[1])

# 2.7 After iterations take mean
# a = a/10
# b = b/10
# c = c/10

# Evaluate Models
# Confusion Matrix
for i in range(3):
    if i == 0:
        cnf_matrix = confusion_matrix(chu_val, LGBM_chu_pred)
    elif i == 1:
        cnf_matrix = confusion_matrix(app_val, LGBM_app_pred)
    else:
        cnf_matrix = confusion_matrix(up_val, LGBM_up_pred)
    class_names = [0, 1]  # name  of classes
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    if i == 0:
        plt.savefig("Conf1.pdf", bbox_inches='tight')
    elif i == 1:
        plt.savefig("Conf2.pdf", bbox_inches='tight')
    else:
        plt.savefig("Conf3.pdf", bbox_inches='tight')
    plt.show()

