FROM continuumio/miniconda3

WORKDIR /app

# Salin conda env
COPY MLProject/telco_churn_preprocessing/conda.yaml .
RUN conda env create -f conda.yaml

# Salin seluruh folder preprocessing
COPY MLProject/telco_churn_preprocessing /app/telco_churn_preprocessing

# Jalankan perintah dalam environment conda
SHELL ["conda", "run", "-n", "mlflow-env", "/bin/bash", "-c"]

CMD ["conda", "run", "-n", "mlflow-env", "python", "telco_churn_preprocessing/modelling.py", "--data_path", "telco_churn_preprocessing/telco_churn_preprocessed.xlsx"]
