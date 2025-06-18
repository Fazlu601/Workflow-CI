FROM continuumio/miniconda3
WORKDIR /app
COPY . /app
RUN conda env create -f MLProject/conda.yaml
SHELL ["conda", "run", "-n", "mlflow-env", "/bin/bash", "-c"]
CMD ["python", "MLProject/modelling.py", "--data_path", "MLProject/telco_churn_preprocessing/telco_churn_preprocessed.xlsx"]
