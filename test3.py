import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

# Fungsi untuk mendapatkan data harga Bitcoin dari CoinGecko API
def get_bitcoin_data():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {
        "vs_currency": "usd",
        "days": "63",  # Ambil data harga selama 63 hari (60 hari lalu plus 3 hari ke depan)
    }
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        raise Exception("Gagal mengambil data harga Bitcoin.")
    
    data = response.json()
    df = pd.DataFrame(data["prices"], columns=["Tanggal", "Harga"])
    df["Tanggal"] = pd.to_datetime(df["Tanggal"], unit="ms")
    return df

# Fungsi untuk menghitung moving average selama 7 hari
def calculate_moving_average(df):
    df["Moving_Avg_7_Days"] = df["Harga"].rolling(window=7).mean()
    return df

# Fungsi untuk menambahkan fitur-fitur tambahan
def add_features(df):
    df["Perubahan_Harian"] = df["Harga"].diff()
    return df

# Fungsi untuk menambahkan fitur tanggal
def add_date_features(df):
    df["Tanggal_Hari"] = df["Tanggal"].dt.day
    df["Tanggal_Hari_Minggu"] = df["Tanggal"].dt.weekday
    return df

# Fungsi untuk melakukan scaling fitur
def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# Fungsi untuk melatih dan menguji model
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    # Membuat model Regresi Linear
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    
    # Membuat model Random Forest Regressor
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    
    # Membuat model XGBoost Regressor
    xgb_model = XGBRegressor(random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    
    return lr_model, rf_model, xgb_model, lr_pred, rf_pred, xgb_pred

# Fungsi untuk menampilkan hasil evaluasi model
def print_evaluation_metrics(model_name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"{model_name} - MSE: {mse:.2f}, MAE: {mae:.2f}")

def main():
    # Mendapatkan data harga Bitcoin
    bitcoin_data = get_bitcoin_data()

    # Menghitung moving average
    bitcoin_data = calculate_moving_average(bitcoin_data)
    
    # Menambahkan fitur-fitur tambahan
    bitcoin_data = add_features(bitcoin_data)
    
    # Menambahkan fitur tanggal
    bitcoin_data = add_date_features(bitcoin_data)
    
    # Persiapan data untuk pemodelan
    X = bitcoin_data[["Harga", "Perubahan_Harian", "Tanggal_Hari", "Tanggal_Hari_Minggu"]]
    y = bitcoin_data["Harga"].shift(-1).dropna()
    
    # Imputasi nilai yang hilang
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)
    
    # Memotong 3 hari terakhir untuk pengujian
    split_index = len(y) - 3
    X_train, X_test, y_train, y_test = train_test_split(X[:split_index], y[:split_index], test_size=0.2, random_state=42)
    
    # Scaling fitur
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    
    # Melatih dan menguji model
    lr_model, rf_model, xgb_model, lr_pred, rf_pred, xgb_pred = train_and_evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Menampilkan hasil evaluasi model
    print("Hasil Evaluasi Model:")
    print_evaluation_metrics("Regresi Linear", y_test, lr_pred)
    print_evaluation_metrics("Random Forest", y_test, rf_pred)
    print_evaluation_metrics("XGBoost", y_test, xgb_pred)
    
    # Prediksi harga Bitcoin untuk 3 hari ke depan
    last_date = bitcoin_data["Tanggal"].iloc[-1]
    next_dates = [last_date + timedelta(days=i) for i in range(1, 4)]
    next_features = np.array([bitcoin_data[["Harga", "Perubahan_Harian", "Tanggal_Hari", "Tanggal_Hari_Minggu"]].iloc[-1]])
    
    # Scaling fitur untuk prediksi
    scaler = StandardScaler()
    next_features_scaled = scaler.fit_transform(next_features)
    
    # Prediksi menggunakan model XGBoost
    next_prices_xgb = [xgb_model.predict(next_features_scaled)[0] for _ in range(3)]
    
    # Menampilkan prediksi harga Bitcoin untuk 3 hari ke depan
    print("\nPrediksi Harga Bitcoin untuk 3 Hari ke Depan:")
    for i, date in enumerate(next_dates):
        print(f"{date.strftime('%Y-%m-%d')}: ${next_prices_xgb[i]:.2f}")
    
    # Visualisasi hasil dalam grafik garis
    plt.figure(figsize=(12, 6))
    
    # Line chart untuk harga asli Bitcoin
    plt.plot(bitcoin_data["Tanggal"][-len(y_test):], y_test.values, label="Harga Asli", linestyle="-")
    
    # Line chart untuk hasil prediksi
    plt.plot(bitcoin_data["Tanggal"][-len(y_test):], lr_pred, label="Prediksi Regresi Linear", linestyle="--")
    plt.plot(bitcoin_data["Tanggal"][-len(y_test):], rf_pred, label="Prediksi Random Forest", linestyle="-.")
    plt.plot(bitcoin_data["Tanggal"][-len(y_test):], xgb_pred, label="Prediksi XGBoost", linestyle=":")
    
    # Menampilkan pred
    # Menampilkan prediksi harga Bitcoin untuk 3 hari ke depan
    plt.plot(next_dates, next_prices_xgb, label="Prediksi XGBoost (3 Hari ke Depan)", linestyle="-.", marker="o")
    
    plt.legend()
    plt.xlabel("Tanggal")
    plt.ylabel("Harga Bitcoin (USD)")
    plt.title("Prediksi Harga Bitcoin dengan Berbagai Model Regresi")
    plt.grid(True)
    
    # Menambahkan garis vertikal untuk memisahkan data pelatihan dan pengujian
    plt.axvline(x=bitcoin_data["Tanggal"].iloc[len(bitcoin_data) - len(y_test)], color='gray', linestyle='--', linewidth=1)
    
    plt.show()

if __name__ == "__main__":
    main()
