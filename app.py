import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import pickle

# Load the model
RFR_model = pickle.load(open('Mtech_RF_model.pkl', 'rb'))
# print(type(RFR_model))

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)
# print('lblencooooooood',label_encoders.keys())

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Define feature labels (no need to specify encodings manually)
feature_labels = {
    'assembly': {'Imported', 'Local'},
    'body': {'Compact sedan', 'Double Cabin', 'Mini Van', 'Truck', 'Hatchback', 'Crossover', 'Pick Up', 'Sedan', 'SUV', 'Van', 'Off-Road Vehicles', 'High Roof', 'Micro Van', 'MPV', 'Station Wagon', 'Compact SUV', 'Single Cabin', 'Coupe', 'Convertible', 'Mini Vehicles'},
    'make': {'MG', 'Proton', 'BAIC', 'Tesla', 'Mercedes', 'Nissan', 'Hyundai', 'BMW', 'Changan', 'Mazda', 'Range', 'Toyota', 'Daihatsu', 'Prince', 'Daewoo', 'Jeep', 'Datsun', 'FAW', 'Porsche', 'Isuzu', 'Audi', 'Subaru', 'United', 'Mitsubishi', 'KIA', 'Chery', 'Land', 'Peugeot', 'Chevrolet', 'DFSK', 'Volkswagen', 'Honda', 'Lexus', 'Haval', 'Suzuki', 'Ford'},
    'model': {'Fit', 'Ek', 'Juke', 'Camry', 'Wrangler', 'CT200h', 'D-Max', 'Hijet', 'Joy', 'Serena', 'Copen', 'AD', 'Sunny', 'IST', 'Probox', 'Zest', 'H6', 'Bolan', 'RX8', 'Yaris', 'Aqua', 'CJ', 'Pajero', 'Sienta', 'Hustler', 'Wagon', 'Hilux', 'Cayenne', 'iQ', 'Bluebird', '3', 'Xbee', 'Life', 'Shehzore', 'Estima', 'C-HR', 'Rav4', 'Kizashi', 'Saga', 'RX', 'March', 'Passo', 'Lancer', 'Stonic', 'Acty', 'Stream', 'X-PV', 'N', 'Khyber', 'Tanto', 'Exclusive', 'Corolla', 'Santro', 'Sonata', 'Spectra', 'Bravo', 'Elantra', 'F', 'APV', 'Mirage', 'V2', 'Accord', 'Vitara', 'Hiace', 'ZS', 'Potohar', 'Avanza', 'CR-V', 'MR', 'Classic', 'Insight', 'Sirius', 'Move', 'A6', 'H-100', 'Glory', 'Duet', 'Grand', 'Fortuner', 'Cuore', 'City', 'BR-V', 'Rush', 'Jolion', 'Vitz', 'Flair', 'Beetle', 'Otti', 'Terios', 'I', 'Prius', 'Freed', 'M9', 'Sportage', 'Pleo', 'Tank', 'Wingroad', 'Moco', 'Starlet', 'Galant', 'Mark', 'Mega', 'BJ40', 'X5', 'Ciaz', 'Succeed', 'Voxy', 'e-tron', 'Every', 'Stella', 'Model', 'Harrier', 'Civic', 'Optra', 'A4', 'Pride', 'Crown', '7', 'Dayz', 'Alsvin', 'A3', 'Jimny', 'ISIS', 'Wish', 'Sorento', 'Cx3', 'Platz', 'Picanto', 'Aygo', 'Pino', 'X70', 'Ravi', 'Prado', 'Premio', 'Liana', 'Pearl', 'LX', 'Grace', 'Panamera', '2008', 'M', 'Tiida', 'HS', 'Tiggo', 'Belta', 'Rvr', 'Corona', '5', 'Esse', 'Fj', 'Patrol', 'Boon', 'X1', 'Roox', 'Pixis', 'Mira', 'Note', 'Clipper', 'Taft', 'K07', 'Karvaan', 'Charade', '120', 'Baleno', 'Raize', 'Benz', 'Swift', 'Vamos', 'HR-V', 'Carrier', 'Allion', 'QQ', 'Oshan', 'Rover', 'Cast', 'Excel', 'Tucson', 'Mehran', 'Noah', 'Q7', 'Roomy', 'Airwave', 'Margalla', 'Surf', 'FX', 'Racer', 'EK', 'Alto', 'Vezel', 'Spacia', 'Land', 'Q3', 'Carry', 'Alpha', 'Cross', 'Carol', 'Cultus', 'Jade', 'CR-Z', 'Spike', 'A5', 'Rocky', 'Kei'},
    'transmission':  {'Automatic', 'Manual'},
    'color': {'maroon', 'navy', 'red', 'purple', 'turquoise', 'blue', 'gold', 'titanium', 'beige', 'gray', 'graphite', 'black', 'green', 'pink', 'steel', 'brown', 'bronze', 'yellow', 'silver', 'magenta', 'orange', 'burgundy', 'indigo', 'white'},
    'registered': {'Nowshera', 'Chakwal', 'Mansehra', 'Bannu', 'Lahore', 'Abbottabad', 'Gujranwala', 'Quetta', 'Kashmir', 'Bhakkar', 'Sindh', 'Sargodha', 'D.G.Khan', 'Un-Registered', 'Sahiwal', 'Gilgit', 'Attock', 'Mian Wali', 'Dera ismail khan', 'Bahawalpur', 'Sheikhupura', 'Rawalpindi', 'Mirpur A.K.', 'Sukkur', 'Rahim Yar Khan', 'Multan', 'Kasur', 'Mardan', 'Peshawar', 'Jhelum', 'Hyderabad', 'Haripur', 'Jhang', 'Rawalakot', 'Swat', 'Faisalabad', 'Vehari', 'Bahawalnagar', 'Kohat', 'Sialkot', 'Gujrat', 'Muzaffarabad', 'Okara', 'Khanewal', 'Islamabad', 'Karachi', 'Punjab'},
    'year': None,
    'mileage': None,
    'engine': None
}

# Define feature transformation function (with automatic encoding)
def transform_features(data):
    
    for feature in label_encoders.keys():
        if feature in data.columns:  # Ensure feature exists in new data
            data[feature] = label_encoders[feature].transform(data[feature])

    # print('bef:',data)
    # Apply scaling to all features
    data = scaler.transform(data)
    # print('aft:',data)
    # Reshape data into input vector format
    input_data = np.array([[data[0, 0], data[0, 1], data[0, 2], data[0, 3], data[0, 4],
                             data[0, 5], data[0, 6], data[0, 7], data[0, 8], data[0, 9]]],
                           dtype=object)
    return input_data

# Create the Streamlit app (rest of the code remains the same)
st.title("Car Price Prediction")

# Get user input for features
assembly = st.selectbox("Assembly", options=feature_labels['assembly'])
body = st.selectbox("Body", options=feature_labels['body'])
make = st.selectbox("Make", options=feature_labels['make'])
model = st.selectbox("Model", options=feature_labels['model'])
year = st.number_input("Year", min_value=1991, max_value=2020)
engine = st.number_input("Engine")
transmission = st.selectbox("Transmission", options=feature_labels['transmission'])
color = st.selectbox("Color", options=feature_labels['color'])
registered = st.selectbox("Registered", options=feature_labels['registered'])
mileage = st.number_input("Mileage")


# Collect input data into a DataFrame
data = pd.DataFrame({
    'assembly': [assembly],
    'body': [body],
    'make': [make],
    'model': [model],
    'year': [year],
    'engine': [engine],
    'transmission': [transmission],
    'color': [color],
    'registered': [registered],
    'mileage': [mileage]
})
print('data:',data)
# Transform features
input_data = transform_features(data)
# print('data:',data)
print('scaled data:',input_data)
# Predict price
if st.button("Predict"):
    prediction = RFR_model.predict(input_data)
    st.write("Predicted price:", prediction[0])
