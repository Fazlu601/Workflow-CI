FROM continuumio/miniconda3

WORKDIR /app

COPY MLProject /app/MLProject

WORKDIR /app/MLProject

RUN conda env create -f conda.yaml

# Aktifkan conda environment secara eksplisit untuk semua perintah berikut
SHELL ["conda", "run", "-n", "mlflow-env", "/bin/bash", "-c"]

# Gunakan CMD dengan bash -c agar environment aktif
CMD ["bash", "-c", "conda run -n mlflow-env python modelling.py --data_path telco_churn_preprocessing/telco_churn_preprocessed.xlsx"]
