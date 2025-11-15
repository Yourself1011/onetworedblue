import modal
from pathlib import Path

weights = modal.Volume.from_name("weights", create_if_missing=True)
WEIGHTS_DIR = Path("/weights")

app = modal.App("onetworedblue")

image = modal.Image.debian_slim(  # define dependencies
    python_version="3.11"
).pip_install("torch==2.5.1", "numpy==2.1.3", "datasets")

with image.imports():  # set up common imports
    import torch
    from .parseevaluations import fen_to_tensor, getData, loadDb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Here, we define the config as a dictionary so that we can re-use it here
# and later, when we attach the profiler. We want to make sure the profiler is in the same environment!

config = {"gpu": "a10g", "image": image, "volumes": {WEIGHTS_DIR: weights}}


def feedforward(
    board,
    w1,
    b1,
    w2,
    b2,
):
    return (
        torch.clamp(
            torch.concat(
                (
                    torch.flatten(board) @ w1 + b1,
                    torch.flatten(torch.flip(board, [1])) @ w1 + b1,
                )
            ),
            0,
            1,
        )
        ** 2
        @ w2
        + b2
    )


def save(w1, b1, w2, b2, optim):
    obj = {
        "weights": {"w1": w1, "b1": b1, "w2": w2, "b2": b2},
        "optim": optim.state_dict(),
    }
    torch.save(obj, WEIGHTS_DIR / "save.pth")


def load():
    try:
        obj = torch.load(WEIGHTS_DIR / "save.pth")
        w1 = obj["weights"]["w1"]
        b1 = obj["weights"]["b1"]
        w2 = obj["weights"]["w2"]
        b2 = obj["weights"]["b2"]
        optim = obj["weights"]["optim"]
    except Exception as e:
        print(e)
        HIDDEN_WIDTH = 1024
        w1 = torch.empty((12 * 64, HIDDEN_WIDTH), requires_grad=True, device=device)
        b1 = torch.zeros(HIDDEN_WIDTH, requires_grad=True, device=device)
        w2 = torch.empty((HIDDEN_WIDTH * 2, 1), requires_grad=True, device=device)
        b2 = torch.zeros(1, requires_grad=True, device=device)
        optim = torch.optim.AdamW([w1, b1, w2, b2], lr=3e-3)
    return w1, b1, w2, b2, optim


@app.function(**config)
def train():
    dataset = loadDb()
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    w1, b1, w2, b2, optim = load()
    board = torch.randint(0, 2, (12, 8, 8), dtype=torch.float32, device=device)

    with torch.no_grad():
        w1.uniform_(-0.001, 0.001)
        w2.uniform_(-0.001, 0.001)

    totalLoss = 0
    n = 1000
    for i in range(50000):
        optim.zero_grad()
        if not i % n:
            save(w1, b1, w2, b2, optim)
            print(i, "loss:", totalLoss / n)
            totalLoss = 0

        evaluation = getData(dataset)
        board = fen_to_tensor(evaluation["fen"], device)

        output = feedforward(board, w1, b1, w2, b2)
        target = evaluation["cp"] / 100
        loss = (target - output) ** 2
        totalLoss += loss
        loss.backward()
        optim.step()

    output = feedforward(board, w1, b1, w2, b2)
    print(output)


@app.local_entrypoint()
def main():
    train.remote()
    print("wenis")
