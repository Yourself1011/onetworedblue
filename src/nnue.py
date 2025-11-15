import modal

app = modal.App("onetworedblue")

image = modal.Image.debian_slim(  # define dependencies
    python_version="3.11"
).pip_install("torch==2.5.1", "numpy==2.1.3")

with image.imports():  # set up common imports
    import torch
    from torch import int32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Here, we define the config as a dictionary so that we can re-use it here
# and later, when we attach the profiler. We want to make sure the profiler is in the same environment!

config = {"gpu": "a10g", "image": image}


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
                    torch.flatten(torch.flip(board, (0, 1))) @ w1 + b1,
                )
            ),
            0,
            1,
        )
        @ w2
        + b2
    )


@app.function(**config)
def train():
    HIDDEN_WIDTH = 1024
    w1 = torch.empty((12 * 64, HIDDEN_WIDTH), requires_grad=True, device=device)
    b1 = torch.zeros(HIDDEN_WIDTH, requires_grad=True, device=device)
    w2 = torch.empty((HIDDEN_WIDTH * 2, 1), requires_grad=True, device=device)
    b2 = torch.zeros(1, requires_grad=True, device=device)
    board = torch.randint(0, 2, (12, 8, 8), dtype=torch.float32, device=device)

    with torch.no_grad():
        w1.uniform_(-0.001, 0.001)
        w2.uniform_(-0.001, 0.001)

    optim = torch.optim.AdamW([w1, b1, w2, b2])

    for i in range(1000):
        optim.zero_grad()
        if not i % 1000:
            print(i)
        output = feedforward(board, w1, b1, w2, b2)
        loss = (1 - output) ** 2
        loss.backward()
        optim.step()

    output = feedforward(board, w1, b1, w2, b2)
    print(output)


@app.local_entrypoint()
def main():
    train.remote()
    print("wenis")
