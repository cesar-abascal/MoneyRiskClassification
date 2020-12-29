# ====================================================================================================
# Application: Risk Analysis using SVM
# File: main.py - Main application file
# Author: Cesar Abascal - cesar.abascal@gmail.com
# Data: 29/12/2020
# Obs.: Based on Eduardo Rocha class
# ====================================================================================================



# Import streamlit lib
import streamlit as st



# FUNCTIONS -------------------------------------------------------------------------------------------

# Load dataset
@st.cache
def get_data():
    # Import lib
    import pandas as pd

    # Return dataframe
    return pd.read_csv("dataset.csv")


# Train SVM model
def train_SVM_model(dataframe):
    # Import libs
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    dataframe = dataframe.drop(columns='id_cliente') # Drop client id
    
    # Split data from labels
    X = dataframe.iloc[:,:-1].values
    Y = dataframe.iloc[:,-1].values

    # Resizing Data - Using StandardScaler
    sc = StandardScaler()
    X_resized = sc.fit_transform(X)

    # Machine training
    mp_svm = SVC(kernel='linear', gamma=1e-5, C=10, random_state=7)
    mp_svm = mp_svm.fit(X_resized, Y)

    # returns the trained machine
    return mp_svm



# MAIN ------------------------------------------------------------------------------------------------

# Read the dataset and create a dataframe
data = get_data()

# Train SVM model
model = train_SVM_model(data)

# Streamlit sidebar --------------------
# Define sidebar sub-header
st.sidebar.subheader("Apresente os dados dos clientes para Análise de Risco")

# Reads data entered by the user. Uses the averages of each data as placeholder
default_rate = st.sidebar.number_input("Índice de Inadimplência", value=data.indice_inad.mean())
registration_notes = st.sidebar.number_input("Anotações Cadastrais", value=data.anot_cadastrais.mean())
income_classification = st.sidebar.number_input("Classificação da Renda", value=data.class_renda.mean())
account_balance = st.sidebar.number_input("Saldo de Contas", value=data.saldo_contas.mean())

# Sets the button
btn_predict = st.sidebar.button("Realizar Predição de Risco")

# Show risk prediction if the button is clicked
if(btn_predict):
    # Classify based on model
    result = model.predict([[default_rate, registration_notes, income_classification, account_balance]])
    risk = str(result[0]).split('_')
    
    # Show the classification result
    st.sidebar.subheader("O risco previsto é: " + risk[1].upper())
 
# Streamlit body -----------------------
# Define streamlit title
st.title("Previsão de risco para concessão de empréstimos á clientes")

# Define streamlit subtitle
st.markdown("Aplicativo Web utilizado para classificar utilizando SVM, e exibir o 'Risco Cliente' durante concessões de empréstimos.")

# Show dataset used on model
st.subheader("Dataset utilizado no modelo")

# Dataset default variables to show
defaultcols = ["anot_cadastrais", "indice_inad", "class_renda", "saldo_contas"]

# Define variables
cols = st.multiselect("Atributos", data.columns.tolist(), default=defaultcols)

# Shows entire dataset based on selected filter
st.dataframe(data[cols])
