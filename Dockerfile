FROM continuumio/miniconda3

WORKDIR /app

COPY MLProject/conda.yaml .

RUN conda env create -f conda.yaml
SHELL ["conda", "run", "-n", "mlflow-env", "/bin/bash", "-c"]

CMD ["conda", "run", "-n", "mlflow-env", "python", "MLProject/modelling.py", "--data_path", "MLProject/telco_churn_preprocessed.xlsx"]
