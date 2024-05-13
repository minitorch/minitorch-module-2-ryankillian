"""
Be sure you have minitorch installed in you Virtual Env.
>>> pip install -Ue .
"""

import minitorch


def RParam(*shape):
    r = 2 * (minitorch.rand(shape) - 0.5)
    return minitorch.Parameter(r)


class Network(minitorch.Module):
    def __init__(self, hidden_layers):
        super().__init__()

        # Submodules
        self.layer1 = Linear(2, hidden_layers)
        self.layer2 = Linear(hidden_layers, hidden_layers)
        self.layer3 = Linear(hidden_layers, 1)

    def forward(self, x: minitorch.Tensor):
        x = self.layer1.forward(x).relu()
        x = self.layer2.forward(x).relu()
        return self.layer3.forward(x).sigmoid()


class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)
        self.out_size = out_size

    def forward(self, x: minitorch.Tensor) -> minitorch.Tensor:
        # Ensure the input dimensions match the expected input size
        assert (
            x.shape[-1] == self.weights.value.shape[0]
        ), "Input size must match weight size"

        # Reshape weights to prepare for matrix multiplication by adding a batch dimension
        reshaped_weights = self.weights.value.view(1, *self.weights.value.shape)

        # Reshape input tensor by adding an output dimension
        reshaped_input = x.view(*x.shape, 1)

        # Perform element-wise multiplication and then sum across the input dimension
        # to perform a dot product equivalent for each sample in the batch
        batch_product = reshaped_weights * reshaped_input
        summed_product = batch_product.sum(dim=1).contiguous()

        # Reshape the summed product to match the output dimension
        # output_shape = (x.shape[0], self.out_size)  # Number of samples by output size
        reshaped_output = summed_product.view(x.shape[0], self.out_size)

        # Add bias to each output in the batch
        # Bias is reshaped to match the batch output shape (1, out_size) for broadcasting
        final_output = reshaped_output + self.bias.value.view(1, self.out_size)

        return final_output


def default_log_fn(epoch, total_loss, correct, losses):
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


class TensorTrain:
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers)

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x]))

    def run_many(self, X):
        return self.model.forward(minitorch.tensor(X))

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):

        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        X = minitorch.tensor(data.X)
        y = minitorch.tensor(data.y)

        losses = []
        for epoch in range(1, self.max_epochs + 1):
            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            # Forward
            out = self.model.forward(X).view(data.N)
            prob = (out * y) + (out - 1.0) * (y - 1.0)

            loss = -prob.log()
            (loss / data.N).sum().view(1).backward()
            total_loss = loss.sum().view(1)[0]
            losses.append(total_loss)

            # Update
            optim.step()

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                y2 = minitorch.tensor(data.y)
                correct = int(((out.detach() > 0.5) == y2).sum()[0])
                log_fn(epoch, total_loss, correct, losses)


if __name__ == "__main__":
    PTS = 50
    HIDDEN = 3
    RATE = 0.5
    data = minitorch.datasets["Simple"](PTS)
    TensorTrain(HIDDEN).train(data, RATE)
