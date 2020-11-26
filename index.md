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
![](DraggedImage.png)
Method2：One-hot encoding（dummy variables）★better
Create a new column for each unique category in a categorical variable. Each observation recieves a 1 in the column for its corresponding category and a 0 in all other new columns.
缺：变量level过多会占用大量内存  → 解决：[dimension reduction](https://www.analyticsvidhya.com/blog/2015/07/dimension-reduction-methods/) （[PCA](http://www.cs.otago.ac.nz/cosc453/student_tutorials/principal_components.pdf)）
优：The value assigned to each of the categories is random and does not reflect any inherent aspect of the category.(无内涵标签以外的任何信息)
![](DraggedImage-1.png)

* For categorical variable `dtype == object` with 2 unique categories 对只有两个level的分类变量 * label encoding* —Scikit-Learn `LabelEncoder` 
	* For categorical variable with more than 2 unique categories * one-hot encoding*—pandas `get_dummies(df)`
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

# Exploratory Data Analysis (EDA)-探索性数据分析
## 分析因变量分布状况-y Column
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
![](DraggedImage-2.png)
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
![](DraggedImage-3.png)![](DraggedImage-4.png)
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
### 模型训练
```python
from sklearn.linear_model import LogisticRegression

# Make the model with the specified regularization parameter
log_reg = LogisticRegression(C = 0.0001)
#lower the regularization parameter, C, which controls the amount of overfitting (a lower value should decrease overfitting)

# Train on the training data
train = df.drop(columns = ['y'])#这里的train是不带标签的训练集
train_labels = df['y']
log_reg.fit(train, train_labels)
```
### 模型预测
```python
# Make predictions
# Make sure to select the second column only
log_reg_pred = log_reg.predict_proba(test)[:, 1]
```
This returns an **m x 2** array where **m** is the number of observations. The first column is the probability of the target being 0 and the second column is the probability of the target being 1 (so for a single row, the two columns must sum to 1).在这里我们选择了第二列（y==1）
# Model improvement
## random forest

# 模型评价
## ROC & AUC
random guessing on a classification task will score a 0.5 
[Can AUC-ROC be between 0-0.5?](https://stats.stackexchange.com/questions/266387/can-auc-roc-be-between-0-0-5)
# Tricks
```python
#当y是0-1变量（1代表positive时），直接计算y的均值就可以得出阳性率
df['y'].mean()
```
