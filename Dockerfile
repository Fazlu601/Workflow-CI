FROM continuumio/miniconda3

WORKDIR /app

# Salin seluruh isi MLProject (pastikan relative to build context)
COPY MLProject /app/MLProject

# Ganti directory kerja ke dalam folder isi project kamu
WORKDIR /app/MLProject/telco_churn_preprocessing

# Buat environment dari conda.yaml
RUN conda env create -f conda.yaml

# Gunakan shell dari environment
SHELL ["conda", "run", "-n", "mlflow-env", "/bin/bash", "-c"]

# Jalankan skrip modelling
CMD ["conda", "run", "-n", "mlflow-env", "python", "modelling.py", "--data_path", "telco_churn_preprocessed.xlsx"]
