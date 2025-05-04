# import pandas as pd
# import numpy as np
# import scipy as sp
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn import metrics

# # %matplotlib inline
# data = pd.read_excel("advertisements_data.xlsx", engine="openpyxl")

# data.isnull().sum()
# names = ['total_price','district','price','area','year','rooms','elevator','parking','warehouse','balcony','unit_per_floor','house_document','wc','renovated','floor_material','cooler','heater','floor_num','total_floor','id','adid','direction','title','hot_water_supplier','description','latitude','longitude', 'map_zone','mobile', 'published_at', 'created_at']

# # Remove commas and convert to numeric for 'rooms' and 'total_price'
# data['total_price'] = data['total_price'].astype(str).str.replace(',', '').astype(float)
# data['price'] = data['price'].astype(str).str.replace(',', '').astype(float)
# data['area'] = pd.to_numeric(data['area'], errors='coerce')  # Convert 'rooms' to numeric
# data['year'] = pd.to_numeric(data['year'], errors='coerce')  # Convert 'rooms' to numeric
# data['rooms'] = pd.to_numeric(data['rooms'], errors='coerce')  # Convert 'rooms' to numeric

# # Drop rows with NaN values after conversion
# data = data.dropna(subset=['price','area','year','rooms'])

# c=['district','area','year','rooms']
# df=data[c]
# df=pd.get_dummies(df,columns=['area'], drop_first=True)
# y=data['price']

# df=data[names]

# df = df.replace(',', '', regex=True)  # Remove commas
# df = df.apply(pd.to_numeric, errors='coerce')
# correlations= df.corr()
# fig=plt.figure()
# ax=fig.add_subplot(111)
# cax=ax.matshow(correlations,vmin=-1,vmax=1)
# fig.colorbar(cax)
# ticks = np.arange(len(names))  # Ensure ticks match the number of labels
# ax.set_xticks(ticks)
# ax.set_yticks(ticks)
# ax.set_xticklabels(names, rotation=90)  # Rotate labels for better readability
# ax.set_yticklabels(names)


# x_train,x_test,y_train,y_test=train_test_split(df,y,train_size=0.8,random_state=42)

# x_train.head()
# reg=LinearRegression()
# reg.fit(x_train,y_train)
# print('Coefficients: \n', reg.coef_)



# # plt.figure(figsize=(20, 12)) # prevent to gather to gether the data on top of the chart
# # sns.stripplot(x='area', y='price', data=data, size=5)

# # plt.xlabel('Rooms')
# # plt.ylabel('Price')
# # plt.title('Price vs Rooms')

# # Boxplot
# # sns.stripplot(x='price', y='area', data=data, size=5, jitter=True, alpha=0.5)

# # sns.boxplot(x='rooms', y='price', data=data)
# # sns.stripplot(x='rooms', y='price',data=data, size=5)

# # plt.xticks(rotation=90) 
# plt.show()

# # print(data.columns)


# # print( )


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer  # Importing imputer
from sklearn import metrics

# %matplotlib inline
data = pd.read_excel("advertisements_data.xlsx", engine="openpyxl")

data.isnull().sum()

names = ['total_price','district','price','area','year','rooms','elevator','parking','warehouse','balcony','unit_per_floor','house_document','wc','renovated','floor_material','cooler','heater','floor_num','total_floor','id','adid','direction','title','hot_water_supplier','description','latitude','longitude', 'map_zone','mobile', 'published_at', 'created_at']

# Remove commas and convert to numeric for 'rooms' and 'total_price'
data['total_price'] = data['total_price'].astype(str).str.replace(',', '').astype(float)
data['price'] = data['price'].astype(str).str.replace(',', '').astype(float)
data['area'] = pd.to_numeric(data['area'], errors='coerce')  # Convert 'rooms' to numeric
data['year'] = pd.to_numeric(data['year'], errors='coerce')  # Convert 'rooms' to numeric
data['rooms'] = pd.to_numeric(data['rooms'], errors='coerce')  # Convert 'rooms' to numeric

# Drop rows with NaN values after conversion
data = data.dropna(subset=['price', 'area', 'year', 'rooms'])

# Prepare your features and target
c=['district','area','year','rooms']
df = data[c]

# Convert categorical 'district' column to numeric (e.g., one-hot encoding)
df = pd.get_dummies(df, columns=['district', 'area'], drop_first=True)

# Target variable
y = data['price']

# Imputation for numeric columns
imputer = SimpleImputer(strategy='median')  # Impute with median for numeric data
df = imputer.fit_transform(df)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(df, y, train_size=0.8, random_state=42)

# Create and train the linear regression model
reg = LinearRegression()
reg.fit(x_train, y_train)

# Print the coefficients
print('Coefficients: \n', reg.coef_)

# Evaluate the model (Optional)
y_pred = reg.predict(x_test)
print('Mean Squared Error: ', metrics.mean_squared_error(y_test, y_pred))


new_house_data = {
    "district": "tehranvila",
    "total_floor": 5,
    "floor_num": 2,
    "area": 72,
    "rooms": 2,
    "year": 1400,
    "unit_per_floor": 2,
    "elevator": True,  # Add missing 'elevator'
    "parking": True,   # Add missing 'parking'
    "warehouse": True, # Add missing 'warehouse'
    "balcony": True,   # Add missing 'balcony'
    "wc": "سرویس بهداشتی ایرانی و فرنگی",
    "cooler": "سرمایش کولر آبی",
    "hot_water_supplier": "تأمین‌کننده آب گرم پکیج",
    "heater": "گرمایش شوفاژ",
    "floor_material": "جنس کف سرامیک",
    "direction": "جنوبی",
}

# Convert to DataFrame
new_house_df = pd.DataFrame([new_house_data])

# Clean boolean-like columns in the new house data
for col in boolean_columns:
    if col in new_house_df.columns:
        new_house_df[col] = new_house_df[col].astype(str)  # Convert to string
        new_house_df[col] = new_house_df[col].str.upper()  # Convert 'TRUE'/'FALSE' to uppercase
        new_house_df[col] = new_house_df[col].replace({'TRUE': True, 'FALSE': False})  # Replace with boolean

# Encode categorical features using saved encoders
for col in categorical_cols:
    if col in new_house_df.columns:
        le = label_encoders[col]
        # Ensure that all categories from training are available in the new data
        
        new_house_df[col] = le.transform(new_house_df[col].astype(str))
        
# Ensure new_house_df contains all necessary columns
missing_cols = [col for col in numeric_cols if col not in new_house_df.columns]
for col in missing_cols:
    new_house_df[col] = np.nan  # Set missing columns to NaN


# Ensure feature order matches the training data (df)
new_house_df = new_house_df[numeric_cols]

# Normalize numerical features using the trained scaler
# new_house_df[numeric_cols] = scaler.transform(new_house_df[numeric_cols])



expected_cols = X_train.columns

# Reindex new_house_df to ensure it has all the features in the correct order.
# Missing columns will be filled with NaN.
new_house_df = new_house_df.reindex(columns=expected_cols, fill_value=np.nan)

# Now, scale only the numeric columns
new_house_df[numeric_cols] = scaler.transform(new_house_df[numeric_cols])



# Predict price (using the trained model)
predicted_price = model.predict(new_house_df)

print(f"💰 Predicted Price: {predicted_price[0]}")

