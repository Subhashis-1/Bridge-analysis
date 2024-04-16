# main.py
from data_generation import generate_data
from data_preprocessing import preprocess_data
from model_training import train_model
from anomaly_detection import detect_anomalies

def main():
    # Generate and preprocess data
    data = generate_data(num_days=60)  # Increase to 60 days for more data
    processed_data = preprocess_data(data)

    # Train the model
    model = train_model(processed_data)

    # Detect anomalies
    anomalies = detect_anomalies(processed_data)
    print("Detected anomalies:")
    print(anomalies[anomalies['anomaly'] == True])

if __name__ == "__main__":
    main()
