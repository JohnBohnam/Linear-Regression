import numpy as np

class Loss:
    def __init__(self):
        pass

    def forward(self, y, y_pred, weights):
        pass

    def backward(self, x, y, y_pred, weights):
        pass


class MSE(Loss):
    def __init__(self):
        super().__init__()

    def forward(self, y, y_pred, weights):
        return np.mean((y - y_pred) ** 2, axis=0)

    def backward(self, x, y, y_pred, weights):
        y = y.reshape(-1, )
        y_pred = y_pred.reshape(-1, )
        return np.dot(x.T, 2 * (y_pred - y)) / x.shape[-1]


class MAE(Loss):
    def __init__(self):
        super().__init__()

    def forward(self, y, y_pred, weights):
        y = y.reshape(-1, )
        y_pred = y_pred.reshape(-1, )
        return np.mean(np.abs(y - y_pred), axis=0)

    def backward(self, x, y, y_pred, weights):
        y = y.reshape(-1, )
        y_pred = y_pred.reshape(-1, )
        return np.dot(x.T, np.sign(y_pred - y)) / x.shape[0]


class L2(Loss):
    def __init__(self, loss: Loss, alpha):
        super().__init__()
        self.loss = loss
        self.alpha = alpha

    def forward(self, y, y_pred, weights):
        return self.loss.forward(y, y_pred, weights) + self.alpha * np.sum(weights ** 2)

    def backward(self, x, y, y_pred, weights):
        return self.loss.backward(x, y, y_pred, weights) + 2 * self.alpha * weights


class L1(Loss):
    def __init__(self, loss: Loss, alpha):
        super().__init__()
        self.loss = loss
        self.alpha = alpha

    def forward(self, y, y_pred, weights):
        return self.loss.forward(y, y_pred, weights) + self.alpha * np.sum(np.abs(weights))

    def backward(self, x, y, y_pred, weights):
        return self.loss.backward(x, y, y_pred, weights) + self.alpha * np.sign(weights)


class LinearModel:
    def __init__(self, input_size, output_size, loss_f: Loss = MSE(), start_weights=None):
        self.input_size = input_size
        self.output_size = output_size
        # self.weights = np.random.randn(input_size, output_size)
        if start_weights is not None:
            self.weights = start_weights

        else:
            self.weights = np.zeros((input_size, output_size))
        self.loss_f = loss_f
        self.step = None

    def forward(self, x):
        return np.dot(x, self.weights)

    def backward(self, x, y, y_pred, optim=False, batch_size=None):
        if batch_size is not None:
            # randomly sample batch_size samples
            idx = np.random.choice(x.shape[0], batch_size, replace=False)
            x = x[idx]
            y = y[idx]
            y_pred = y_pred[idx]
        base_loss = self.loss_f.forward(y, y_pred, self.weights)
        base_weights = self.weights.copy()

        grad = self.loss_f.backward(x, y, y_pred, self.weights)
        self.weights -= self.step * grad
        act_loss = self.loss_f.forward(y, self.forward(x), self.weights)

        if not optim:
            return act_loss

        min_step = 1e-50
        while act_loss > base_loss and self.step > min_step:
            # find out if the step is too big
            self.step /= 2.0
            self.weights = base_weights - self.step * grad
            act_loss = self.loss_f.forward(y, self.forward(x), self.weights)

        prev_loss = act_loss
        prev_weights = self.weights.copy()
        while act_loss <= prev_loss:
            # find out if the step is too small
            self.step *= 2.0
            prev_loss = act_loss
            prev_weights = self.weights.copy()
            self.weights = base_weights - self.step * grad
            act_loss = self.loss_f.forward(y, self.forward(x), self.weights)

        # now binsearch between prev_weights and self.weights
        eps = self.step / 1e4
        coef_prev = 0
        coef_next = 1
        while coef_next - coef_prev > eps:
            coef_mid = (coef_prev + coef_next) / 2
            self.weights = prev_weights * (1 - coef_mid) + self.weights * coef_mid
            if act_loss < prev_loss:
                coef_next = coef_mid
            else:
                coef_prev = coef_mid

        loss = self.loss_f.forward(y, self.forward(x), self.weights)
        return loss

    def fit_gradient_descent(self,
                             x, y, val_x, val_y,
                             lr,
                             epochs,
                             optim=False,
                             mov_avg_n=5,
                             mov_avg_N=15,
                             min_improvement=1e-3,
                             min_epochs=0,
                             verbose=True,
                             batch_size=None):
        self.weights = self.weights.reshape(-1)
        self.step = lr
        loss_hist = []
        val_loss_hist = []
        best_loss = self.evaluate(val_x, val_y)
        best_params = self.weights.copy()

        loss_hist.append(self.evaluate(x, y))
        val_loss_hist.append(best_loss)

        for i in range(epochs):
            y_pred = self.forward(x)
            loss = self.backward(x, y, y_pred, optim, batch_size=batch_size)
            valid_loss = self.evaluate(val_x, val_y)

            if valid_loss < best_loss:
                best_loss = valid_loss
                best_params = self.weights.copy()

            loss_hist.append(loss)
            val_loss_hist.append(valid_loss)
            if i >= mov_avg_N:
                loc_avg = np.mean(val_loss_hist[-mov_avg_n:])
                bigger_avg = np.mean(val_loss_hist[-mov_avg_N:])
                if i >= min_epochs and abs(loc_avg / bigger_avg - 1) < min_improvement:
                    if verbose:
                        print(f'Early stopping at epoch {i + 1}')
                        print(f'Best validation loss: {best_loss}')
                    self.weights = best_params
                    break

            if verbose and i % 100 == 0:
                print(f'Epoch {i + 1}/{epochs} Loss: {loss}')
                print(f'Epoch {i + 1}/{epochs} Validation Loss: {valid_loss}')

        return loss_hist, val_loss_hist

    def fit_analytical(self, x, y):
        if isinstance(self.loss_f, MSE):
            xTx_inv = np.linalg.inv(np.dot(x.T, x))
            xTy = np.dot(x.T, y)
            self.weights = np.dot(xTx_inv, xTy)
        elif isinstance(self.loss_f, L2) and isinstance(self.loss_f.loss, MSE):
            xTx_inv = np.linalg.inv(np.dot(x.T, x) + self.loss_f.alpha * np.eye(x.shape[1]))
            xTy = np.dot(x.T, y)
            self.weights = np.dot(xTx_inv, xTy)
        else:
            raise NotImplementedError('Analytical solution not implemented for this loss function')

    def evaluate(self, x, y):
        y_pred = self.forward(x)
        return self.loss_f.forward(y, y_pred, self.weights)

    def summary(self, x, y):
        print(f'Weights: {self.weights}')
        print(f'Loss: {self.evaluate(x, y)}')
