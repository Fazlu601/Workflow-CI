FROM continuumio/miniconda3

# Buat folder kerja di container
WORKDIR /app

# Copy seluruh isi MLProject ke dalam container
COPY MLProject /app/MLProject

# Pindah ke folder MLProject tempat conda.yaml berada
WORKDIR /app/MLProject

# Buat environment dari conda.yaml
RUN conda env create -f conda.yaml

# Gunakan environment tersebut untuk semua perintah berikutnya
SHELL ["conda", "run", "-n", "mlflow-env", "/bin/bash", "-c"]

# Jalankan script modelling.py dan arahkan ke file xlsx yang ada di subfolder
CMD ["python", "modelling.py", "--data_path", "telco_churn_preprocessing/telco_churn_preprocessed.xlsx"]
