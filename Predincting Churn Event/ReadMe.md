###  **Churn Prediction For Sprint**

Imagine you're working with Sprint, one of the biggest telecom companies in the USA. They're really keen on figuring out how many customers might
decide to leave them in the coming months. Luckily, they've got a bunch of past data about when customers have left before, as well as info
about who these customers are, what they've bought, and other things like that. 
So, if you were in charge of predicting customer churn, how would you go about using machine learning to make a good guess about which  
customers might leave? What steps would you take to create a machine learning model that can predict if someone's going to leave or not?
    

### **Solution**
What is customer churn? Customer churn refers to the number of customers who stopped using a product or a companies services in a given 
period of time. Understanding the reasons behind customer churn is vital to sustaining a healthy business. In this case the churn event 
would involve the customer cancling his/her subscriptions.

### **Step 1: Data Collection and Understanding**
To build a churn prediction model for Sprint, you need a dataset that includes various data attributes (features) about the customers.
- Gather historical customer data.
- Perform exploratory data analysis (EDA) to understand the dataset.interactions with the company, and their historical churn status.
In this project,  generated my own data using mockaroo.com 
 ''' python
  import pandas as pd

#Read the dataset
data_df = pd.read_csv("path/sprint_data.csv")
#Get overview of the data
print(sprint_data.head())
sprint_data.info()
sprint_data.describe()
'''

### **Step 2: Data Preprocessing**
In this section, we’ll gain more insights and convert the data into a data representation suitable for various machine learning algorithms.
Data Cleaning:
Handle missing values and outliers. Clean the data to ensure it's suitable for modeling.
''' python
# The customerID column isnt useful as the feature is used for identification of customers.
sprint_data.drop(["id"],axis=1,inplace = True)

# Encode categorical features
encoded_sprint_data= pd.get_dummies (sprint_data[['location', 'gender', 'contract', 'complaints', 'plandetails']], drop_first=True)

encoded_sprint_data['location_Anchorage'] = encoded_sprint_data['location_Anchorage'].astype(int)
#Join the two data frames to form one dataframe
int_data = sprint_data [['tenure', 'no_of_support_calls', 'monthlycharges', 'churn']]
sprintdata_final= pd.concat([int_data,encoded_sprint_data], axis=1)
'''


### **step 3: Model Selection and Training**
In this project i will opt to Choose XGBoost as the prediction model. 
- Train the model on the training data.
''' python
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,recall_score,f1_score,precision_score,rec_auc_score

x =sprintdata_final.drop('churn', axis=1)
y =sprintdata_final['churn']

#split data into train and test tests
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

#Initialize and train the model
model=xgb.XGBClassifier() 
model.fit(x_train,y_train)

#make predictions
ypred= model.predict(xtest)
    '''


### **step 4: Model Evaluation**
   - Evaluate the model's performance using metrics like accuracy, precision, recall
  '''python
   accuracy = accuracy_score(y_test,ypred)
   recall = recall_score(y_test,ypred)
   f1 = f1_score(y_test,ypred)
   precision = precision_score(y_test,ypred)
    '''

###   **step 5: Displaying results**
'''python
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score,recall_score,f1_score,precision_score
    
print('confusion matrix :',confusion_matrix(y_test,ypred))
print('classification report :',classification_report(y_test,ypred))
print('accuracy :',round(accuracy_score(y_test,ypred),2))
print('recall :',round(recall_score(y_test,ypred),2))
print('f1 :',round(f1_score(y_test,ypred),2))
print('precision:',round(precision_score(y_test,ypred),2))
print()
,,,

