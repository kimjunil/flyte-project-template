# -*- coding: utf-8 -*-
import torch
import math
from flytekit import task, workflow, Resources
import numpy as np
import pandas as pd
from model.polynomial3 import Polynomial3


@task
def generate_data(n: int) -> pd.DataFrame:
    x = np.linspace(-math.pi, math.pi, n)
    y = np.sin(x)

    return pd.DataFrame({"x": x, "y": y})


@task(requests=Resources(cpu="4", mem="4Gi"))
def train(df: pd.DataFrame):

    x, y = df["x"].values, df["y"].values
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()

    model = Polynomial3()

    criterion = torch.nn.MSELoss(reduction="sum")
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)
    for t in range(2000):
        y_pred = model(x)

        loss = criterion(y_pred, y)
        if t % 100 == 99:
            print(t, loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Result: {model.string()}")


@workflow
def train_workflow(n: int = 2000):
    train(df=generate_data(n=n))


if __name__ == "__main__":
    train_workflow(n=2000)
