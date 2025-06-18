FROM continuumio/miniconda3

# Set working directory di dalam container
WORKDIR /app

# Salin seluruh isi folder MLProject ke dalam container
COPY MLProject /app/MLProject

# Set working directory ke folder preprocessing
WORKDIR /app/MLProject/telco_churn_preprocessing

# Buat environment conda dari file conda.yaml
RUN conda env create -f conda.yaml

# Aktifkan environment untuk perintah berikutnya
SHELL ["conda", "run", "-n", "mlflow-env", "/bin/bash", "-c"]

# Jalankan script modelling.py
CMD ["bash", "-c", "python modelling.py --data_path telco_churn_preprocessed.xlsx"]
