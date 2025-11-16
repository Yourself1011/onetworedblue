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
    import math
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
    batchSize = board.size(0)
    return (
        torch.clamp(
            torch.concat(
                (
                    torch.reshape(board, (batchSize, -1)) @ w1 + b1,
                    torch.reshape(torch.flip(board, [2, 3]), (batchSize, -1)) @ w1 + b1,
                ),
                dim=1,
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
        "w1": w1.detach().cpu(),
        "b1": b1.detach().cpu(),
        "w2": w2.detach().cpu(),
        "b2": b2.detach().cpu(),
        "optim": optim.state_dict(),
    }
    torch.save(obj, WEIGHTS_DIR / "savetmp.pth")
    (WEIGHTS_DIR / "savetmp.pth").rename(WEIGHTS_DIR / "save.pth")


def load():
    try:
        # raise Exception
        obj = torch.load(WEIGHTS_DIR / "save.pth", map_location=device)
        w1 = obj["w1"].to(device).detach().clone().requires_grad_(True)
        b1 = obj["b1"].to(device).detach().clone().requires_grad_(True)
        w2 = obj["w2"].to(device).detach().clone().requires_grad_(True)
        b2 = obj["b2"].to(device).detach().clone().requires_grad_(True)
        optim = torch.optim.AdamW([w1, b1, w2, b2], lr=1e-3, weight_decay=1e-5)
        optim.load_state_dict(obj["optim"])
    except Exception as e:
        print("there was an error:")
        print(e)
        HIDDEN_WIDTH = 1024
        w1 = torch.empty((12 * 64, HIDDEN_WIDTH), requires_grad=True, device=device)
        b1 = torch.zeros(HIDDEN_WIDTH, requires_grad=True, device=device)
        w2 = torch.empty((HIDDEN_WIDTH * 2, 1), requires_grad=True, device=device)
        b2 = torch.zeros(1, requires_grad=True, device=device)
        optim = torch.optim.AdamW([w1, b1, w2, b2], lr=1e-3, weight_decay=1e-5)

        with torch.no_grad():
            torch.nn.init.xavier_uniform_(w1)
            torch.nn.init.xavier_uniform_(w2)
    return w1, b1, w2, b2, optim


@app.function(**config, timeout=24 * 60 * 60)
def train():
    dataset = loadDb()
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    w1, b1, w2, b2, optim = load()

    board = torch.randint(0, 2, (12, 8, 8), dtype=torch.float32, device=device)

    # print((torch.flatten(board) @ w1 + b1).max())
    totalLoss = 0
    n = 64
    batchSize = 64

    # torch.manual_seed(0)
    # boards = torch.stack(
    #     [
    #         torch.randint(
    #             0,
    #             2,
    #             (12, 8, 8),
    #             dtype=torch.float32,
    #             device=device,
    #         )
    #         for _ in range(batchSize)
    #     ]
    # )
    for i in range(1000000):
        optim.zero_grad()

        evaluations = [getData(dataset) for _ in range(batchSize)]
        boards = torch.stack(
            [fen_to_tensor(evaluations[i]["fen"], device) for i in range(batchSize)]
        )

        outputs = feedforward(boards, w1, b1, w2, b2)
        targets = torch.tensor(
            [math.atan(evaluations[i]["cp"] / 300) for i in range(batchSize)],
            dtype=torch.float32,
            device=device,
        ).view(batchSize, 1)
        # targets = torch.tensor(
        #     [i for i in range(batchSize)],
        #     dtype=torch.float32,
        #     device=device,
        # ).view(batchSize, 1)
        loss = torch.mean((targets - outputs) ** 2)
        totalLoss += loss.item()
        loss.backward()
        optim.step()

        if not i % n:
            save(w1, b1, w2, b2, optim)
            print(i * n, "loss:", math.sqrt(totalLoss / n))
            # print(targets.max().item(), targets.min().item())
            # print(outputs.max().item(), outputs.min().item())
            totalLoss = 0

    output = feedforward(board, w1, b1, w2, b2)
    print(output)


@app.local_entrypoint()
def main():
    train.remote()
    print("wenis")
