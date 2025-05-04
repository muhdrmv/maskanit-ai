# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import OneHotEncoder

# # --- Load and Clean the Dataset ---
# df = pd.read_excel("advertisements_data.xlsx", engine="openpyxl")
# df = df.drop(columns=['id', 'adid', 'mobile', 'published_at', 'created_at', 'description', 'title', 'token_id', 'latitude', 'longitude', 'map_zone'])


# def one_hot_encode_columns(df, columns):

#     results = {}

#     for col in columns:

#         df[col] = df[col].fillna(0)

#         unique_vals = df[col].unique()
#         label_map = {cat: idx for idx, cat in enumerate(unique_vals)}

#         df[col] = df[col].map(label_map).astype(str)

#         one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
#         encoded = one_hot_encoder.fit_transform(df[[col]])

#         results[col] = {
#             "encoded_lists": encoded.tolist(),
#             "encoding_classes": list(label_map.keys())
#         }

#     return results

# columns_to_encode = [
#     "house_document", "unit_per_floor", "floor_material",
#     "renovated", "heater", "cooler", "hot_water_supplier", 
#     "direction", "district", "rooms", "floor_num", "total_floor"
# ]

# encoded_dict = one_hot_encode_columns(df, columns_to_encode)

# # house_document_encoded_lists = encoded_dict["house_document"]["encoded_lists"]
# # house_document_classes       = encoded_dict["house_document"]["encoding_classes"]

# # print(len(encoded_dict["rooms"]["encoding_classes"]))

# encoded_arrays = [
#     np.array(encoded_dict[col]["encoded_lists"]) 
#     for col in columns_to_encode
# ]
# final_encode = np.hstack(encoded_arrays)

# print(len(final_encode[0]))


# column_slices = {}
# start_col = 0
# for col, arr in zip(columns_to_encode, encoded_arrays):
#     # arr.shape[1] is how many one-hot columns for this feature
#     width = arr.shape[1]
#     end_col = start_col + width
#     # store the slice (start_col, end_col)
#     column_slices[col] = (start_col, end_col)
#     start_col = end_col

# print(column_slices)



















# import pandas as pd
# import numpy as np

# df = pd.read_excel("advertisements_data2.xlsx", engine='openpyxl')
# df = df.drop(columns=['id', 'adid', 'mobile', 'published_at', 'created_at', 'description', 'title', 'token_id', 'latitude', 'longitude', 'map_zone'])


# df['price'] = df['price'].replace({',': ''}, regex=True).astype(float)

# # drop rows with price smaller than 10,000,000
# df = df[df['price'] >= 10000000]
# # drop rows with price greater than 300,000,000
# df = df[df['price'] <= 300000000]
# df['price'] = df['price'].astype(int)

# df['year'] = df['year'].replace('qbl z 1370', 1360)
# df['year'] = df['year'].astype(int)


# print(f"number of rows after filters : {len(df)}")


# from sklearn.preprocessing import OneHotEncoder

# def one_hot_encode_columns(df, columns):

#     results = {}

#     for col in columns:

#         df[col] = df[col].fillna(0)

#         unique_vals = df[col].unique()
#         label_map = {cat: idx for idx, cat in enumerate(unique_vals)}

#         df[col] = df[col].map(label_map).astype(str)

#         one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
#         encoded = one_hot_encoder.fit_transform(df[[col]])

#         results[col] = {
#             "encoded_lists": encoded.tolist(),
#             "encoding_classes": list(label_map.keys())
#         }

#     return results


# columns_to_encode = [
#     "house_document", "unit_per_floor", "wc", "floor_material",
#     "renovated", "heater", "cooler", "hot_water_supplier",
#     "direction", "district", "rooms", "floor_num", "total_floor"
# ]

# encoded_dict = one_hot_encode_columns(df, columns_to_encode)

# print(df['price'].min())
# print(df['price'].max())




# import matplotlib.pyplot as plt

# number_of_bins = 10

# price_full_range = df['price'].max() - df['price'].min()
# price_bin_range = price_full_range / number_of_bins

# price_bins = []
# for round in range(number_of_bins):
#     price_bins.append(
#         {
#             "min": int((round)*price_bin_range+df['price'].min())+1,
#             "max": int((round+1)*price_bin_range+df['price'].min())
#         }
#     )

# counter_bins = [0] * number_of_bins

# for index, row in df.iterrows():
#     for index, bin in enumerate(price_bins) :
#         if row['price']>=bin["min"] and row['price']<=bin["max"]:
#             counter_bins[index] += 1

# x_positions = range(len(counter_bins))

# # Create a bar plot
# plt.bar(x_positions, counter_bins)

# # Add labels and title
# plt.xlabel('Index')
# plt.ylabel('Value')
# plt.title('Bar Plot of Array')

# # Show the plot
# plt.show()







# def encode_output_by_bins(bins, price):
#     encoding = [0] * len(bins)
#     for index, bin in enumerate(bins) :
#         if row['price']>=bin["min"] and row['price']<=bin["max"]:
#             encoding[index] = 1
#             return encoding
#         elif row['price']<bin["min"]:
#             encoding[0] = 1
#             return encoding




# def normalization(df, value):
#     return (value - df.min()) / (df.max() - df.min())

# def true_false_to_zero_one(value):
#     if value == "TRUE":
#         return 1
#     else:
#         return 0




# inputs = []
# outputs = []

# for index in range(len(df)):
#     row = df.iloc[index]

#     normalized_area = normalization(df['area'], row['area'])
#     normalized_year = normalization(df['year'], row['year'])

#     input = [normalized_area, normalized_year, true_false_to_zero_one(row['elevator']), true_false_to_zero_one(row['parking']), true_false_to_zero_one(row['warehouse']), true_false_to_zero_one(row['balcony'])]

#     for key in encoded_dict:
#         input += encoded_dict[key]["encoded_lists"][index]

#     inputs.append(input)

#     outputs.append(encode_output_by_bins(price_bins, row['price']))



# import random

# indices = list(range(len(inputs)))
# random.shuffle(indices)

# inputs_shuffled = [inputs[i] for i in indices]
# outputs_shuffled = [outputs[i] for i in indices]

# test_element_number = 100

# inputs_training = np.array(inputs_shuffled[test_element_number:])
# inputs_testing = np.array(inputs_shuffled[:test_element_number])

# outputs_training = np.array(outputs_shuffled[test_element_number:])
# outputs_testing = np.array(outputs_shuffled[:test_element_number])




# # input shape
# input_shape = len(inputs[0])
# print(f"input_shape : {input_shape}")

# # output shape
# output_shape = len(outputs[0])
# print(f"output_shape : {output_shape}")






# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.layers import Input


# # Define the neural network
# def create_neural_network(input_shape=347, output_classes=10):
#     model = Sequential([
#         Input(shape=(input_shape,)),  # <--- More standard way
#         Dense(256, activation='relu'),
#         BatchNormalization(),
#         Dropout(0.3),

#         Dense(128, activation='relu'),
#         BatchNormalization(),
#         Dropout(0.2),

#         Dense(64, activation='relu'),
#         BatchNormalization(),
#         Dropout(0.2),

#         Dense(output_classes, activation='softmax')
#     ])

#     model.compile(
#         optimizer='adam',
#         loss='categorical_crossentropy',
#         metrics=['accuracy']
#     )
#     return model

# # model = create_neural_network(input_shape=348, output_classes=10)
# model = create_neural_network(input_shape=input_shape, output_classes=10)


# # model.summary()

# history = model.fit(
#     inputs_training,
#     outputs_training,
#     epochs=10,
#     batch_size=32,
#     validation_split=0.2
# )





# predictions = model.predict(inputs_testing)
# predicted_classes = np.argmax(predictions, axis=1)

# corrects = 0

# for index, prediction in enumerate(predicted_classes):
#   indices = np.where(outputs_testing[index] == 1)

#   if prediction == indices[0][0]:
#     corrects+=1

# print(f"accuracy : {corrects/len(predicted_classes)}")






# ==== YOUR ORIGINAL CODE ====

import pandas as pd
import numpy as np

df = pd.read_excel("advertisements_data2.xlsx", engine='openpyxl')
df = df.drop(columns=['id', 'adid', 'mobile', 'published_at', 'created_at', 'description', 'title', 'token_id', 'latitude', 'longitude', 'map_zone'])

df['price'] = df['price'].replace({',': ''}, regex=True).astype(float)

df = df[df['price'] >= 10000000]
df = df[df['price'] <= 300000000]
df['price'] = df['price'].astype(int)

df['year'] = df['year'].replace('qbl z 1370', 1360)
df['year'] = df['year'].astype(int)

print(f"number of rows after filters : {len(df)}")

from sklearn.preprocessing import OneHotEncoder

def one_hot_encode_columns(df, columns):
    results = {}
    for col in columns:
        df[col] = df[col].fillna(0)
        unique_vals = df[col].unique()
        label_map = {cat: idx for idx, cat in enumerate(unique_vals)}
        df[col] = df[col].map(label_map).astype(str)

        one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded = one_hot_encoder.fit_transform(df[[col]])

        results[col] = {
            "encoded_lists": encoded.tolist(),
            "encoding_classes": list(label_map.keys())
        }

    return results

columns_to_encode = [
    "house_document", "unit_per_floor", "wc", "floor_material",
    "renovated", "heater", "cooler", "hot_water_supplier",
    "direction", "district", "rooms", "floor_num", "total_floor"
]

encoded_dict = one_hot_encode_columns(df, columns_to_encode)

print(df['price'].min())
print(df['price'].max())

import matplotlib.pyplot as plt

number_of_bins = 10
price_full_range = df['price'].max() - df['price'].min()
price_bin_range = price_full_range / number_of_bins

price_bins = []
for round in range(number_of_bins):
    price_bins.append(
        {
            "min": int((round)*price_bin_range+df['price'].min())+1,
            "max": int((round+1)*price_bin_range+df['price'].min())
        }
    )

counter_bins = [0] * number_of_bins

for index, row in df.iterrows():
    for index, bin in enumerate(price_bins) :
        if row['price']>=bin["min"] and row['price']<=bin["max"]:
            counter_bins[index] += 1

x_positions = range(len(counter_bins))

plt.bar(x_positions, counter_bins)
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Bar Plot of Array')
plt.show()

def encode_output_by_bins(bins, price):
    encoding = [0] * len(bins)
    for index, bin in enumerate(bins) :
        if price >= bin["min"] and price <= bin["max"]:
            encoding[index] = 1
            return encoding
        elif price < bin["min"]:
            encoding[0] = 1
            return encoding

def normalization(df, value):
    return (value - df.min()) / (df.max() - df.min())

def true_false_to_zero_one(value):
    if value == "TRUE":
        return 1
    else:
        return 0

inputs = []
outputs = []

for index in range(len(df)):
    row = df.iloc[index]

    normalized_area = normalization(df['area'], row['area'])
    normalized_year = normalization(df['year'], row['year'])

    input = [normalized_area, normalized_year, true_false_to_zero_one(row['elevator']), true_false_to_zero_one(row['parking']), true_false_to_zero_one(row['warehouse']), true_false_to_zero_one(row['balcony'])]

    for key in encoded_dict:
        input += encoded_dict[key]["encoded_lists"][index]

    inputs.append(input)

    outputs.append(encode_output_by_bins(price_bins, row['price']))

import random

indices = list(range(len(inputs)))
random.shuffle(indices)

inputs_shuffled = [inputs[i] for i in indices]
outputs_shuffled = [outputs[i] for i in indices]

test_element_number = 100

inputs_training = np.array(inputs_shuffled[test_element_number:])
inputs_testing = np.array(inputs_shuffled[:test_element_number])

outputs_training = np.array(outputs_shuffled[test_element_number:])
outputs_testing = np.array(outputs_shuffled[:test_element_number])

input_shape = len(inputs[0])
print(f"input_shape : {input_shape}")

output_shape = len(outputs[0])
print(f"output_shape : {output_shape}")

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input

def create_neural_network(input_shape=347, output_classes=10):
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(output_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model = create_neural_network(input_shape=input_shape, output_classes=10)

history = model.fit(
    inputs_training,
    outputs_training,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)

predictions = model.predict(inputs_testing)
predicted_classes = np.argmax(predictions, axis=1)

corrects = 0
for index, prediction in enumerate(predicted_classes):
    indices = np.where(outputs_testing[index] == 1)
    if prediction == indices[0][0]:
        corrects+=1

print(f"difference accuracy : {corrects/len(predicted_classes)}")

# ==== 🔍 FEATURE IMPORTANCE SECTION ====

from copy import deepcopy

def get_feature_importance(model, inputs, outputs, encoded_dict, columns_to_encode, original_input_length):
    from copy import deepcopy
    import numpy as np

    outputs = np.array(outputs)
    base_predictions = model.predict(np.array(inputs))
    base_classes = np.argmax(base_predictions, axis=1)

    correct_base = 0
    for i, pred in enumerate(base_predictions):
        true_class_indices = np.where(outputs[i] == 1)[0]
        if len(true_class_indices) == 0:
            continue  # Skip bad samples
        if np.argmax(pred) == true_class_indices[0]:
            correct_base += 1

    base_accuracy = correct_base / len(base_classes)
    importance_scores = {}

    index_offset = 6  # First 6 features are not from one-hot
    for col in columns_to_encode:
        col_encoding_length = len(encoded_dict[col]["encoding_classes"])
        col_indices = list(range(index_offset, index_offset + col_encoding_length))
        index_offset += col_encoding_length

        inputs_shuffled = deepcopy(inputs)
        for i in range(len(inputs_shuffled)):
            temp_vals = [inputs_shuffled[i][j] for j in col_indices]
            random.shuffle(temp_vals)
            for j, val in zip(col_indices, temp_vals):
                inputs_shuffled[i][j] = val

        predictions = model.predict(np.array(inputs_shuffled))
        predicted_classes = np.argmax(predictions, axis=1)

        corrects = 0
        for index, prediction in enumerate(predicted_classes):
            true_class_indices = np.where(outputs[index] == 1)[0]
            if len(true_class_indices) == 0:
                continue
            if prediction == true_class_indices[0]:
                corrects += 1

        accuracy = corrects / len(predicted_classes)
        accuracy_drop = base_accuracy - accuracy
        importance_scores[col] = accuracy_drop

    return importance_scores

# ==== 📊 PLOT THE IMPORTANCE ====

importance_scores = get_feature_importance(
    model,
    inputs_testing.tolist(),
    outputs_testing.tolist(),
    encoded_dict,
    columns_to_encode,
    input_shape
)

import matplotlib.pyplot as plt

sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
features = [x[0] for x in sorted_features]
scores = [x[1] * 100 for x in sorted_features]  # percentage drop

plt.figure(figsize=(12, 6))
plt.bar(features, scores, color='skyblue')
plt.ylabel("Accuracy Drop (%)")
plt.title("Feature Importance (Permutation-based)")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()
