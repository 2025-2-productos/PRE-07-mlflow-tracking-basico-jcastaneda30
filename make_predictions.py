import mlflow
import pandas as pd

FILE_PATH = "data/winequality-red.csv"

df = pd.read_csv(FILE_PATH)
y = df["quality"]
X = df.drop(columns=["quality"])

# Debe verificarse el run_id del modelo que se quiere cargar
# Se puede obtener el run_id desde la interfaz de MLflow

logged_model = "runs:/01d6fd7bb85746ecabec9080f31c7db9/model"
loaded_model = mlflow.pyfunc.load_model(logged_model)
y_pred = loaded_model.predict(X)
print(y_pred)
