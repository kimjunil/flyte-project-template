import torch
import math
from flytekit import task, workflow, Resources
import numpy as np
import pandas as pd
from model.polynomial3 import Polynomial3
import mlflow

@task
def generate_data(n: int) -> pd.DataFrame:
    x = np.linspace(-math.pi, math.pi, n)
    y = np.sin(x)

    df = pd.DataFrame({"x": x, "y": y})

    # log data to MLflow
    df.to_csv("data.csv")
    mlflow.log_artifact("data.csv")

    return pd.DataFrame({"x": x, "y": y})


@task(
    requests=Resources(cpu="4", mem="4Gi")
)
def train(df: pd.DataFrame, rl: float):

    # log paramiters to MLflow
    mlflow.log_param("rl", rl)

    x, y = df["x"].values, df["y"].values
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()

    model = Polynomial3()
    metric_loss = {}

    criterion = torch.nn.MSELoss(reduction="sum")
    optimizer = torch.optim.SGD(model.parameters(), lr=rl)
    for t in range(2000):
        y_pred = model(x)

        loss = criterion(y_pred, y)
        if t % 100 == 99:
            metric_loss[t] = loss.item()
            # log metric to MLflow
            mlflow.log_metric("loss", loss.item(), step=t)
            print(t, loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # log model to MLflow
    mlflow.pytorch.log_model(model, "model")

    print(f"Result: {model.string()}")


@workflow
def train_workflow(n: int = 2000, rl: float = 1e-6):
    
    # set MLflow experiment name
    mlflow.set_experiment("Default")

    # strat mlflow run
    with mlflow.start_run() as run:
        train(df=generate_data(n=n), rl=rl)


if __name__ == "__main__":
    train_workflow(n=2000)
