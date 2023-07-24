
import pandas as pd
import pickle

# load the model
model = pickle.load(open('model/model.pkl', 'rb'))

# define the input data
data =  data = [{
                "Account_Balance": 1,
                "Duration_of_Credit_monthly": 18,
                "Payment_Status_of_Previous_Credit": 4,
                "Purpose": 2,
                "Credit_Amount": 1049,
                "Value_Savings_Stocks":	1,
                "Length_of_current_employment":	2,
                "Instalment_per_cent": 4,
                "Sex_Marital_Status": 2,
                "Guarantors": 1,
                "Duration_in_Current_address": 4,
                "Most_valuable_available_asset": 2,
                "Age_years": 21,
                "Concurrent_Credits": 3,
                "Type_of_apartment": 1,
                "No_of_Credits_at_this_Bank": 1,
                "Occupation": 3,
                "No_of_dependents":	1,
                "Telephone": 1,
                "Foreign_Worker": 1
            }]
#check the type of data



print(type(data))

# convert the input data to a dataframe
# the columns are the keys of the dictionary
input_data = pd.DataFrame(data)

# predict the output
output = model.predict_proba(input_data)

print(output)