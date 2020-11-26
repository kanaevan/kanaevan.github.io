Classification 问题的数据分析流程
 <script type="text/x-mathjax-config">
 MathJax.Hub.Config({tex2jax: {inlineMath:[['$latex','$']]}});
 </script>
 <script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>


```python
# numpy and pandas for data manipulation
import numpy as np
import pandas as pd 

# sklearn preprocessing for dealing with categorical variables
from sklearn.preprocessing import LabelEncoder

# File system manangement
import os

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')

# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns

#缺失值填充
from sklearn.impute import SimpleImputer
```
# Data preparation

## Glimpse of Data 数据预览
```python
# Read in data into a dataframe 
data = pd.read_csv('../input/train_v2.csv')
# Display top of dataframe
data.head()
df.columns
data.shape
data.info()
df.describe()

#切分测试集与训练集
train = df[df['y'].notnull()]
test = df[df['y'].isnull()]
```
## 缺失值处理

1. 寻找缺失值
(1). Looking at the number and percentage of missing values in each column.
```python
# Function to calculate missing values by column# Funct 
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns
```
result overview
```python
# Missing values statistics
missing_values = missing_values_table(df)
missing_values.head(20)
```
2. Imputation 缺失值填充/删除

```python
data.fillna(data.mean(), inplace=True) #用均值填充
data.dropna(inplace=True) #删除有空值的行
```
## Anomalies 异常值处理
Anomalies：由于失误、故障产生的异常数值（比如大于3m的身高或者负数的生日），或者是有效但极端的outlier。
1. 识别
```python
#查看每列的统计信息可对异常值进行初步筛选
df.describe()
df.plot.hist()
#使用切片筛选异常值
df[df['SK_ID_CURR'] > 300000] 
```
2. 处理
(1). 设为缺失值按照缺失值处理
```python
df['SK_ID_CURR'][df['SK_ID_CURR'] > 300000] = np.nan
```
(2). 删除所在行
 
## [Encoding Categorical Variables](https://datascience.stackexchange.com/questions/9443/when-to-use-one-hot-encoding-vs-labelencoder-vs-dictvectorizor) 分类变量编码

Method1：Label encoding
assign each unique category in a categorical variable with an integer. No new columns are created.
缺：内涵了顺序和间隔
优：相对节省空间
![](https://raw.githubusercontent.com/WillKoehrsen/Machine-Learning-Projects/master/label_encoding.png)
Method2：One-hot encoding（dummy variables）★better
Create a new column for each unique category in a categorical variable. Each observation recieves a 1 in the column for its corresponding category and a 0 in all other new columns.
缺：变量level过多会占用大量内存  → 解决：[dimension reduction](https://www.analyticsvidhya.com/blog/2015/07/dimension-reduction-methods/) （[PCA](http://www.cs.otago.ac.nz/cosc453/student_tutorials/principal_components.pdf)）
优：The value assigned to each of the categories is random and does not reflect any inherent aspect of the category.(无内涵标签以外的任何信息)
![](https://raw.githubusercontent.com/WillKoehrsen/Machine-Learning-Projects/master/one_hot_encoding.png).

* For categorical variable `dtype == object` with 2 unique categories 对只有两个level的分类变量 * label encoding* —Scikit-Learn `LabelEncoder` 

* For categorical variable with more than 2 unique categories * one-hot encoding*—pandas `get_dummies(df)`.

```python
# Create a label encoder object
le = LabelEncoder()
le_count = 0

# Iterate through the columns
for col in df:
    if df[col].dtype == 'object':
        # If 2 or fewer unique categories
        if len(list(df[col].unique())) <= 2:
            # Train on the training data
            le.fit(df[col])
            # Transform both training and testing data
            df[col] = le.transform(df[col])
            app_test[col] = le.transform(app_test[col])
            
            # Keep track of how many columns were label encoded
            le_count += 1
            
print('%d columns were label encoded.' % le_count)

# one-hot encoding of categorical variables
df = pd.get_dummies(df)
app_test = pd.get_dummies(app_test)

print('Training Features shape: ', df.shape)
print('Testing Features shape: ', app_test.shape)
```

## Feature Scaling 标准化
```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train = sc.fit_transform(train) #不含标签
test = sc.transform(test)
```
# Exploratory Data Analysis (EDA)-探索性数据分析
## 自变量分布状况-x
```python
plt.figure(figsize=(12,5))
plt.title("Distribution of AMT_CREDIT")
ax = sns.distplot(df[3])
```
## 因变量分布状况-y Column
方法：柱状图 hist - balance or [imbalance](http://www.chioka.in/class-imbalance-problem/)
## Column Types
```
# Number of each type of column 
df.dtypes.value_counts() #返回每种数据类型各有几个变量（列）
# Number of unique classes in each object column
df.select_dtypes('object').apply(pd.Series.nunique, axis = 0) #返回每个分类变量（列）的label 数量
```
## Correlations
### Pearson correlation
We can calculate the [Pearson correlation](http://www.statstutor.ac.uk/resources/uploaded/pearsons.pdf) coefficient between every x and the y using the `.corr` dataframe method.
* .00-.19 “very weak”
* .20-.39 “weak”
* .40-.59 “moderate”
* .60-.79 “strong”
* .80-1.0 “very strong”
```python
#计算相关系数，并按各个自变量x与y的相关系数的值升序排列
correlations = df.corr()['y'].sort_values()
```
### kernel density estimation plot ([KDE](https://en.wikipedia.org/wiki/Kernel_density_estimation))
To visualize the effect of the x on the y. A kernel density estimate plot shows the distribution of a single variable and can be thought of as a smoothed histogram (it is created by computing a kernel, usually a Gaussian, at each data point and then averaging all the individual kernels to develop a single smooth curve). 
[Conceptual Foundations](https://chemicalstatistician.wordpress.com/2013/06/09/exploratory-data-analysis-kernel-density-estimation-in-r-on-ozone-pollution-data-in-new-york-and-ozonopolis/)
用于观察x与y之间的关系（不同y-level下x的概率密度分布情况，类似于平滑的直方图）
Use the __seaborn__ `kdeplot` 
```python
plt.figure(figsize = (10, 8))

# KDE plot of y at level 0
sns.kdeplot(df.loc[df['y'] == 0, 'x_i'], label = 'y == 0')

# KDE plot of y at level 1
sns.kdeplot(df.loc[df['y'] == 1, 'x_i'], label = 'y == 1')

# Labeling of plot
plt.xlabel('x'); plt.ylabel('Density'); plt.title('Distribution of x');
```
![](DraggedImage.png)
** 自动化函数-迭代绘制每个x与y的KDE**
```python
plt.figure(figsize = (10, 12))

# iterate through the sources
for i, source in enumerate(['x_1', 'x_2', 'x_3']):
    
    # create a new subplot for each source
    plt.subplot(3, 1, i + 1)
    # plot y at level 0
    sns.kdeplot(df.loc[df['y'] == 0, source], label = 'y == 0')
    # plot y at level 1
    sns.kdeplot(df.loc[df['y'] == 1, source], label = 'y == 1')
    
    # Label the plots
    plt.title('Distribution of %s by y Value' % source)
    plt.xlabel('%s' % source); plt.ylabel('Density');
    
plt.tight_layout(h_pad = 2.5)
```
### 分箱
```
# x_i information into a separate dataframe
xi_y = df[['y', 'x_i']]
```

```
xi_y
Out[]: 
       1          4
0    1.0  -9.445982
1    0.0   8.477863
2    0.0 -14.356526
3    1.0   9.732738
4    1.0 -15.654541
```

```python
# Bin the age data
xi_y['box'] = pd.cut(df['x_i'], bins = np.linspace(-20, 20, num = 9))
```

```
xi_y
Out[]: 
       1               4
0    1.0   (-10.0, -5.0]
1    0.0     (5.0, 10.0]
2    0.0  (-15.0, -10.0]
3    1.0     (5.0, 10.0]
4    1.0  (-20.0, -15.0]
```

```python
# Group by the bin and calculate averages
xi_y  = xi_y.groupby('x_i').mean()#这里的计算均值，相当于计算每个箱子下y=1的百分比
```

```
xi_y
Out[]: 
                       	y
x_i                       
(-20.0, -15.0]  0.532258
(-15.0, -10.0]  0.505747
(-10.0, -5.0]   0.554545
(-5.0, 0.0]     0.361446
(0.0, 5.0]      0.500000
(5.0, 10.0]     0.504348
(10.0, 15.0]    0.444444
(15.0, 20.0]    0.476923
```
接下来可以画图比较不同箱子下y=1的频率大小
### Heatmap of correlations 热力图
```python
plt.figure(figsize = (8, 6))
cor = df.corr()
# Heatmap of correlations
sns.heatmap(cor,cmap=plt.cm.RdYlBu_r,vmin=-0.25,annot = True, vmax = 0.6)
plt.title('Correlation Heatmap')
```
### [Pairs Plot](https://towardsdatascience.com/visualizing-data-with-pair-plots-in-python-f228cf529166)
A pairs plot allows us to see both distribution of single variables and relationships between two variables. Pair plots are a great method to identify trends for follow-up analysis
```python
# Function to calculate correlation coefficient between two columns
def corr_func(x, y, **kwargs):
    r = np.corrcoef(x, y)[0][1]
    ax = plt.gca()
    ax.annotate("r = {:.2f}".format(r),
                xy=(.2, .8), xycoords=ax.transAxes,
                size = 20)

# Create the pairgrid object
grid = sns.PairGrid(data = plot_data, size = 3, diag_sharey=False,
                    hue = 'y', 
                    vars = [x for x in list(plot_data.columns) if x != 'y'])

# Upper is a scatter plot
grid.map_upper(plt.scatter, alpha = 0.2)

# Diagonal is a histogram
grid.map_diag(sns.kdeplot)

# Bottom is density plot
grid.map_lower(sns.kdeplot, cmap = plt.cm.OrRd_r);

plt.suptitle('Pairs Plot', size = 32, y = 1.05);
```
![](DraggedImage-1.png)![](DraggedImage-2.png)
(第二张图是用随机数画的，也就是变量间完全不相关的情况)
# Feature Engineering

Feature engineering refers to a general process and can involve both feature construction: adding new features from the existing data, and feature selection: choosing only the most important features or other methods of dimensionality reduction. 
## Polynomial features
`sklearn.preprocessing.PolynomialFeatures`[documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)
In this method, we make features that are powers of existing features as well as interaction terms between existing features. 
for $latex x_1,x_2,x_3$ we can create variables $latex x_1^2,x_1^3$ and interaction terms like $latex x_1 \cdot x_2,x_1 \cdot x_2 \cdot x_3$ and so on

The class`sklearn.preprocessing.PolynomialFeatures` creates the polynomials and the interaction terms up to a specified degree. 

## Domain knowledge features
利用与问题相关的专业知识主观进行特征工程


# Model Implementation — Baseline
## Logistic Regression
use `LogisticRegressionfrom` Scikit-Learn [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
**Scikit-Learn modeling syntax**: we first create the model, then we train the model using `.fit` and then we make predictions on the testing data using `.predict_proba`

```python
from sklearn.linear_model import LogisticRegression

# Make the model with the specified regularization parameter
log_reg = LogisticRegression(C = 0.0001)
#lower the regularization parameter, C, which controls the amount of overfitting (a lower value should decrease overfitting)

# Train on the training data
train = df.drop(columns = ['y'])#这里的train是不带标签的训练集
train_labels = df['y']
log_reg.fit(train, train_labels)

# Make predictions
# Make sure to select the second column only
log_reg_pred = log_reg.predict_proba(test)[:, 1]
```
This returns an **m x 2** array where **m** is the number of observations. The first column is the probability of the target being 0 and the second column is the probability of the target being 1 (so for a single row, the two columns must sum to 1).在这里我们选择了第二列（y==1）

## Random forest
> Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks that operate by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean/average prediction (regression) of the individual trees.
```python
from sklearn.ensemble import RandomForestClassifier

# Make the random forest classifier
random_forest = RandomForestClassifier(n_estimators = 100, random_state = 50, verbose = 1, n_jobs = -1)

# Train on the training data
random_forest.fit(train, train_labels) #同上，train无标签，标签单列

# Extract feature importances
feature_importance_values = random_forest.feature_importances_
feature_importances = pd.DataFrame({'feature': features, 'importance': feature_importance_values})

# Make predictions on the test data
predictions = random_forest.predict_proba(test)[:, 1]
```

Model Interpretation: Feature Importances - 特征重要性
> Feature importances are not the most sophisticated method to interpret a model or perform dimensionality reduction, but they let us start to understand what factors our model takes into account when it makes predictions.
```python
def plot_feature_importances(df):
    """
    Plot importances returned by a model. This can work with any measure of
    feature importance provided that higher importance is better. 
    
    Args:
        df (dataframe): feature importances. Must have the features in a column
        called `features` and the importances in a column called `importance
        
    Returns:
        shows a plot of the 15 most importance features
        
        df (dataframe): feature importances sorted by importance (highest to lowest) 
        with a column for normalized importance
        """
    
    # Sort features according to importance
    df = df.sort_values('importance', ascending = False).reset_index()
    
    # Normalize the feature importances to add up to one
    df['importance_normalized'] = df['importance'] / df['importance'].sum()

    # Make a horizontal bar chart of feature importances
    plt.figure(figsize = (10, 6))
    ax = plt.subplot()
    
    # Need to reverse the index to plot most important on top
    ax.barh(list(reversed(list(df.index[:15]))), 
            df['importance_normalized'].head(15), 
            align = 'center', edgecolor = 'k')
    
    # Set the yticks and labels
    ax.set_yticks(list(reversed(list(df.index[:15]))))
    ax.set_yticklabels(df['feature'].head(15))
    
    # Plot labeling
    plt.xlabel('Normalized Importance'); plt.title('Feature Importances')
    plt.show()
    
    return df

# Show the feature importances for the default features
feature_importances_sorted = plot_feature_importances(feature_importances)
```
## Naive Bayes
```python
from sklearn.naive_bayes import GaussianNB
naive = GaussianNB()
naive.fit(train,train_labels)
```
## Gradiente Boosting Classification
```python
from xgboost import XGBClassifier
gb = XGBClassifier()
gb.fit(train,train_labels)
```
#  Gradient boosting machine
> [A Gentle Introduction to the Gradient Boosting Algorithm for Machine Learning](https://machinelearningmastery.com/gentle-introduction-gradient-boosting-algorithm-machine-learning/)
> Gradient boosting machine: an efficient algorithm for converting relatively poor hypotheses into very good hypotheses
```python
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import gc

def model(features, test_features, encoding = 'ohe', n_folds = 5):
    
    """Train and test a light gradient boosting model using
    cross validation. 
    
    Parameters
    --------
        features (pd.DataFrame): 
            dataframe of training features to use 
            for training a model. Must include the TARGET column.
        test_features (pd.DataFrame): 
            dataframe of testing features to use
            for making predictions with the model. 
        encoding (str, default = 'ohe'): 
            method for encoding categorical variables. Either 'ohe' for one-hot encoding or 'le' for integer label encoding
            n_folds (int, default = 5): number of folds to use for cross validation
        
    Return
    --------
        submission (pd.DataFrame): 
            dataframe with `SK_ID_CURR` and `TARGET` probabilities
            predicted by the model.
        feature_importances (pd.DataFrame): 
            dataframe with the feature importances from the model.
        valid_metrics (pd.DataFrame): 
            dataframe with training and validation metrics (ROC AUC) for each fold and overall.
        
    """
    
    # Extract the ids
    train_ids = features['SK_ID_CURR']
    test_ids = test_features['SK_ID_CURR']
    
    # Extract the labels for training
    labels = features['TARGET']
    
    # Remove the ids and target
    features = features.drop(columns = ['SK_ID_CURR', 'TARGET'])
    test_features = test_features.drop(columns = ['SK_ID_CURR'])
    
    
    # One Hot Encoding
    if encoding == 'ohe':
        features = pd.get_dummies(features)
        test_features = pd.get_dummies(test_features)
        
        # Align the dataframes by the columns
        features, test_features = features.align(test_features, join = 'inner', axis = 1)
        
        # No categorical indices to record
        cat_indices = 'auto'
    
    # Integer label encoding
    elif encoding == 'le':
        
        # Create a label encoder
        label_encoder = LabelEncoder()
        
        # List for storing categorical indices
        cat_indices = []
        
        # Iterate through each column
        for i, col in enumerate(features):
            if features[col].dtype == 'object':
                # Map the categorical features to integers
                features[col] = label_encoder.fit_transform(np.array(features[col].astype(str)).reshape((-1,)))
                test_features[col] = label_encoder.transform(np.array(test_features[col].astype(str)).reshape((-1,)))

                # Record the categorical indices
                cat_indices.append(i)
    
    # Catch error if label encoding scheme is not valid
    else:
        raise ValueError("Encoding must be either 'ohe' or 'le'")
        
    print('Training Data Shape: ', features.shape)
    print('Testing Data Shape: ', test_features.shape)
    
    # Extract feature names
    feature_names = list(features.columns)
    
    # Convert to np arrays
    features = np.array(features)
    test_features = np.array(test_features)
    
    # Create the kfold object
    k_fold = KFold(n_splits = n_folds, shuffle = True, random_state = 50)
    
    # Empty array for feature importances
    feature_importance_values = np.zeros(len(feature_names))
    
    # Empty array for test predictions
    test_predictions = np.zeros(test_features.shape[0])
    
    # Empty array for out of fold validation predictions
    out_of_fold = np.zeros(features.shape[0])
    
    # Lists for recording validation and training scores
    valid_scores = []
    train_scores = []
    
    # Iterate through each fold
    for train_indices, valid_indices in k_fold.split(features):
        
        # Training data for the fold
        train_features, train_labels = features[train_indices], labels[train_indices]
        # Validation data for the fold
        valid_features, valid_labels = features[valid_indices], labels[valid_indices]
        
        # Create the model
        model = lgb.LGBMClassifier(n_estimators=10000, objective = 'binary', 
                                   class_weight = 'balanced', learning_rate = 0.05, 
                                   reg_alpha = 0.1, reg_lambda = 0.1, 
                                   subsample = 0.8, n_jobs = -1, random_state = 50)
        
        # Train the model
        model.fit(train_features, train_labels, eval_metric = 'auc',
                  eval_set = [(valid_features, valid_labels), (train_features, train_labels)],
                  eval_names = ['valid', 'train'], categorical_feature = cat_indices,
                  early_stopping_rounds = 100, verbose = 200)
        
        # Record the best iteration
        best_iteration = model.best_iteration_
        
        # Record the feature importances
        feature_importance_values += model.feature_importances_ / k_fold.n_splits
        
        # Make predictions
        test_predictions += model.predict_proba(test_features, num_iteration = best_iteration)[:, 1] / k_fold.n_splits
        
        # Record the out of fold predictions
        out_of_fold[valid_indices] = model.predict_proba(valid_features, num_iteration = best_iteration)[:, 1]
        
        # Record the best score
        valid_score = model.best_score_['valid']['auc']
        train_score = model.best_score_['train']['auc']
        
        valid_scores.append(valid_score)
        train_scores.append(train_score)
        
        # Clean up memory
        gc.enable()
        del model, train_features, valid_features
        gc.collect()
        
    # Make the submission dataframe
    submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': test_predictions})
    
    # Make the feature importance dataframe
    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})
    
    # Overall validation score
    valid_auc = roc_auc_score(labels, out_of_fold)
    
    # Add the overall scores to the metrics
    valid_scores.append(valid_auc)
    train_scores.append(np.mean(train_scores))
    
    # Needed for creating dataframe of validation scores
    fold_names = list(range(n_folds))
    fold_names.append('overall')
    
    # Dataframe of validation scores
    metrics = pd.DataFrame({'fold': fold_names,
                            'train': train_scores,
                            'valid': valid_scores}) 
    
    return submission, feature_importances, metrics

submission, fi, metrics = model(app_train, app_test)
print('Baseline metrics')
print(metrics)
fi_sorted = plot_feature_importances(fi)
submission.to_csv('baseline_lgb.csv', index = False) #submission
```
## LightGBM
> LightGBM is a gradient boosting framework that uses tree based learning algorithms. It is designed to be distributed and efficient with the following advantages:
> * Faster training speed and higher efficiency.
> * Lower memory usage.
> * Better accuracy.
> * Support of parallel and GPU learning.
> * Capable of handling large-scale data.
[documentation](https://lightgbm.readthedocs.io/en/latest/) /[Parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html) /[Algorithm](https://papers.nips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf) /[知乎](https://zhuanlan.zhihu.com/p/99069186) /[Wiki](https://en.wikipedia.org/wiki/LightGBM)/[CDSN](https://blog.csdn.net/weixin_39807102/article/details/81912566)/ [medium-block](https://medium.com/@pushkarmandot/https-medium-com-pushkarmandot-what-is-lightgbm-how-to-implement-it-how-to-fine-tune-the-parameters-60347819b7fc)
```python
# 01. train set and test set 划分训练集和测试集
train_data = lgb.Dataset(dtrain[predictors],label=dtrain[target],feature_name=list(dtrain[predictors].columns), categorical_feature=dummies)

test_data = lgb.Dataset(dtest[predictors],label=dtest[target],feature_name=list(dtest[predictors].columns), categorical_feature=dummies)

# 02. parameters 参数设置
param = {
    'max_depth':6,
    'num_leaves':64,
    'learning_rate':0.03,
    'scale_pos_weight':1,
    'num_threads':40,
    'objective':'binary',
    'bagging_fraction':0.7,
    'bagging_freq':1,
    'min_sum_hessian_in_leaf':100
}

param['is_unbalance']='true'
param['metric'] = 'auc'

#03. cv and train 自定义cv函数和模型训练
bst=lgb.cv(param,train_data, num_boost_round=1000, nfold=3, early_stopping_rounds=30)

estimators = lgb.train(param,train_data,num_boost_round=len(bst['auc-mean']))

#04. test predict 测试集结果
ypred = estimators.predict(dtest[predictors])
```
二分类
```python
import lightgbm as lgb  
import pandas as pd  
import numpy as np  
import pickle  
from sklearn.metrics import roc_auc_score  
from sklearn.model_selection import train_test_split  

print("Loading Data ... ")  

# 导入数据  
train_x, train_y, test_x = load_data()  

# 用sklearn.cross_validation进行训练数据集划分，这里训练集和交叉验证集比例为7：3，可以自己根据需要设置  
X, val_X, y, val_y = train_test_split(  
    train_x,  
    train_y,  
    test_size=0.05,  
    random_state=1,  
    stratify=train_y # 这里保证分割后y的比例分布与原数据一致  
)  

X_train = X  
y_train = y  
X_test = val_X  
y_test = val_y  

# create dataset for lightgbm  
lgb_train = lgb.Dataset(X_train, y_train)  
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)  
# specify your configurations as a dict  
params = {  
    'boosting_type': 'gbdt',  
    'objective': 'binary',  
    'metric': {'binary_logloss', 'auc'},  #二进制对数损失
    'num_leaves': 5,  
    'max_depth': 6,  
    'min_data_in_leaf': 450,  
    'learning_rate': 0.1,  
    'feature_fraction': 0.9,  
    'bagging_fraction': 0.95,  
    'bagging_freq': 5,  
    'lambda_l1': 1,    
    'lambda_l2': 0.001,  # 越小l2正则程度越高  
    'min_gain_to_split': 0.2,  
    'verbose': 5,  
    'is_unbalance': True  
}  

# train  
print('Start training...')  
gbm = lgb.train(params,  
                lgb_train,  
                num_boost_round=10000,  
                valid_sets=lgb_eval,  
                early_stopping_rounds=500)  

print('Start predicting...')  

preds = gbm.predict(test_x, num_iteration=gbm.best_iteration)  # 输出的是概率结果  

# 导出结果  
threshold = 0.5  
for pred in preds:  
    result = 1 if pred > threshold else 0  

# 导出特征重要性  
importance = gbm.feature_importance()  
names = gbm.feature_name()  
with open('./feature_importance.txt', 'w+') as file:  
    for index, im in enumerate(importance):  
        string = names[index] + ', ' + str(im) + '\n'  
        file.write(string)  
```
多分类
```python
import lightgbm as lgb  
import pandas as pd  
import numpy as np  
import pickle  
from sklearn.metrics import roc_auc_score  
from sklearn.model_selection import train_test_split  

print("Loading Data ... ")  

# 导入数据  
train_x, train_y, test_x = load_data()  

# 用sklearn.cross_validation进行训练数据集划分，这里训练集和交叉验证集比例为7：3，可以自己根据需要设置  
X, val_X, y, val_y = train_test_split(  
    train_x,  
    train_y,  
    test_size=0.05,  
    random_state=1,  
    stratify=train_y ## 这里保证分割后y的比例分布与原数据一致  
)  

X_train = X  
y_train = y  
X_test = val_X  
y_test = val_y  


# create dataset for lightgbm  
lgb_train = lgb.Dataset(X_train, y_train)  
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)  
# specify your configurations as a dict  
params = {  
    'boosting_type': 'gbdt',  
    'objective': 'multiclass',  
    'num_class': 9,  
    'metric': 'multi_error',  
    'num_leaves': 300,  
    'min_data_in_leaf': 100,  
    'learning_rate': 0.01,  
    'feature_fraction': 0.8,  
    'bagging_fraction': 0.8,  
    'bagging_freq': 5,  
    'lambda_l1': 0.4,  
    'lambda_l2': 0.5,  
    'min_gain_to_split': 0.2,  
    'verbose': 5,  
    'is_unbalance': True  
}  

# train  
print('Start training...')  
gbm = lgb.train(params,  
                lgb_train,  
                num_boost_round=10000,  
                valid_sets=lgb_eval,  
                early_stopping_rounds=500)  

print('Start predicting...')  

preds = gbm.predict(test_x, num_iteration=gbm.best_iteration)  # 输出的是概率结果  

# 导出结果  
for pred in preds:  
    result = prediction = int(np.argmax(pred))  

# 导出特征重要性  
importance = gbm.feature_importance()  
names = gbm.feature_name()  
with open('./feature_importance.txt', 'w+') as file:  
    for index, im in enumerate(importance):  
        string = names[index] + ', ' + str(im) + '\n'  
        file.write(string)  
```
official example
```python
import json
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score
path="D:/data/"
print("load data")
df_train=pd.read_csv(path+"regression.train.csv",header=None,sep='\t')
df_test=pd.read_csv(path+"regression.train.csv",header=None,sep='\t')
y_train = df_train[0].values
y_test = df_test[0].values
X_train = df_train.drop(0, axis=1).values
X_test = df_test.drop(0, axis=1).values
# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
# specify your configurations as a dict
params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'l2', 'auc'},
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
        }
print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=20,
                valid_sets=lgb_eval,
                early_stopping_rounds=5)
print('Save model...')
# save model to file
gbm.save_model(path+'lightgbm/model.txt')
print('Start predicting...')
# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
# eval
print(y_pred)
print('The roc of prediction is:', roc_auc_score(y_test, y_pred) )
print('Dump model to JSON...')
# dump model to json (and save to file)
model_json = gbm.dump_model()
with open(path+'lightgbm/model.json', 'w+') as f:
    json.dump(model_json, f, indent=4)
print('Feature names:', gbm.feature_name())
print('Calculate feature importances...')
# feature importances
print('Feature importances:', list(gbm.feature_importance()))
```
# 模型评价
## ROC & AUC
random guessing on a classification task will score a 0.5 
[Can AUC-ROC be between 0-0.5?](https://stats.stackexchange.com/questions/266387/can-auc-roc-be-between-0-0-5)
## Mean absolute error  — k-Fold Cross Validation
```python
# Function to calculate mean absolute error
def cross_val(train, train_labels, model):
    # Applying k-Fold Cross Validation
    from sklearn.model_selection import cross_val_score
    accuracies = cross_val_score(estimator = model, X = train, y = train_labels, cv = 5)
    return accuracies.mean()
```
