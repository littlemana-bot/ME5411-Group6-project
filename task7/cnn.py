"""Task 7-1: a small convolutional network implemented from scratch with NumPy.

Only NumPy is used for the model itself: the convolution forward/backward pass
is written with an im2col + matrix-multiply formulation, pooling keeps its
argmax, and the optimiser is Adam.  Nothing here relies on a deep-learning
framework, which follows the assignment note that students are encouraged to
develop their own code instead of depending on ready-made packages.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Low level helpers
# --------------------------------------------------------------------------- #
def im2col(x: np.ndarray, kh: int, kw: int, pad: int, stride: int = 1):
    """Turn sliding windows into columns: (N,C,H,W) -> (N, C*kh*kw, H_out*W_out)."""
    n, c, h, w = x.shape
    h_out = (h + 2 * pad - kh) // stride + 1
    w_out = (w + 2 * pad - kw) // stride + 1
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    shape = (n, c, kh, kw, h_out, w_out)
    strides = (xp.strides[0], xp.strides[1], xp.strides[2], xp.strides[3],
               stride * xp.strides[2], stride * xp.strides[3])
    windows = np.lib.stride_tricks.as_strided(xp, shape=shape, strides=strides)
    return windows.reshape(n, c * kh * kw, h_out * w_out), h_out, w_out


def col2im(cols: np.ndarray, x_shape, kh: int, kw: int, pad: int, stride: int = 1):
    """Adjoint of :func:`im2col`; accumulates the gradient of every window."""
    n, c, h, w = x_shape
    h_out = (h + 2 * pad - kh) // stride + 1
    w_out = (w + 2 * pad - kw) // stride + 1
    xp = np.zeros((n, c, h + 2 * pad, w + 2 * pad), dtype=cols.dtype)
    cols = cols.reshape(n, c, kh, kw, h_out, w_out)
    for i in range(kh):
        for j in range(kw):
            xp[:, :, i:i + stride * h_out:stride, j:j + stride * w_out:stride] += cols[:, :, i, j]
    return xp[:, :, pad:pad + h, pad:pad + w] if pad else xp


# --------------------------------------------------------------------------- #
# Layers
# --------------------------------------------------------------------------- #
class Conv2D:
    """Valid/same convolution with square kernels (stride 1)."""

    def __init__(self, c_in: int, c_out: int, k: int = 3, pad: int = 1, rng=None, name: str = 'conv'):
        rng = np.random.default_rng(0) if rng is None else rng
        self.k, self.pad, self.c_out, self.name = k, pad, c_out, name
        # He initialisation for ReLU networks.
        scale = np.sqrt(2.0 / (c_in * k * k))
        self.W = (rng.standard_normal((c_out, c_in * k * k)) * scale).astype(np.float32)
        self.b = np.zeros(c_out, dtype=np.float32)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        n, _, h, w = x.shape
        cols, h_out, w_out = im2col(x, self.k, self.k, self.pad)
        self._cache = (cols, x.shape, h_out, w_out)
        out = self.W @ cols                       # (c_out, K) @ (N, K, H_out*W_out)
        return out.reshape(n, self.c_out, h_out, w_out) + self.b[None, :, None, None]

    def backward(self, dout: np.ndarray) -> np.ndarray:
        cols, x_shape, h_out, w_out = self._cache
        dout2 = dout.reshape(dout.shape[0], self.c_out, h_out * w_out)
        # Written in place: the optimiser holds references to these buffers.
        self.dW[...] = np.matmul(dout2, cols.transpose(0, 2, 1)).sum(axis=0)
        self.db[...] = dout2.sum(axis=(0, 2))
        dcols = self.W.T @ dout2                  # (K, c_out) @ (N, c_out, P)
        return col2im(dcols, x_shape, self.k, self.k, self.pad)

    def parameters(self):
        return [('W', self.W, self.dW, True), ('b', self.b, self.db, False)]

    def shapes(self):
        return {'type': 'conv', 'c_out': self.c_out, 'k': self.k, 'pad': self.pad,
                'c_in': self.W.shape[1] // (self.k * self.k)}


class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask

    def parameters(self):
        return []

    def shapes(self):
        return {'type': 'relu'}


class MaxPool2x2:
    """2x2 max pooling with stride 2; odd borders are trimmed."""

    def forward(self, x):
        n, c, h, w = x.shape
        h2, w2 = h // 2, w // 2
        self.in_shape = (n, c, h, w)
        x = x[:, :, :h2 * 2, :w2 * 2]
        blocks = x.reshape(n, c, h2, 2, w2, 2).transpose(0, 1, 2, 4, 3, 5).reshape(n, c, h2, w2, 4)
        self.idx = blocks.argmax(axis=-1)
        self.out = np.take_along_axis(blocks, self.idx[..., None], axis=-1)[..., 0]
        return self.out

    def backward(self, dout):
        n, c, h2, w2 = dout.shape
        blocks = np.zeros((n, c, h2, w2, 4), dtype=dout.dtype)
        np.put_along_axis(blocks, self.idx[..., None], dout[..., None], axis=-1)
        dx = np.zeros(self.in_shape, dtype=dout.dtype)
        dx[:, :, :h2 * 2, :w2 * 2] = blocks.reshape(n, c, h2, w2, 2, 2).transpose(0, 1, 2, 4, 3, 5).reshape(n, c, h2 * 2, w2 * 2)
        return dx

    def parameters(self):
        return []

    def shapes(self):
        return {'type': 'maxpool2'}


class Flatten:
    def forward(self, x):
        self.shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.shape)

    def parameters(self):
        return []

    def shapes(self):
        return {'type': 'flatten'}


class Dense:
    def __init__(self, n_in: int, n_out: int, rng=None, name: str = 'fc'):
        rng = np.random.default_rng(0) if rng is None else rng
        self.name = name
        self.W = (rng.standard_normal((n_in, n_out)) * np.sqrt(2.0 / n_in)).astype(np.float32)
        self.b = np.zeros(n_out, dtype=np.float32)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        self.dW[...] = self.x.T @ dout
        self.db[...] = dout.sum(axis=0)
        return dout @ self.W.T

    def parameters(self):
        return [('W', self.W, self.dW, True), ('b', self.b, self.db, False)]

    def shapes(self):
        return {'type': 'dense', 'n_in': self.W.shape[0], 'n_out': self.W.shape[1]}


class Dropout:
    """Inverted dropout: activations are scaled during training only."""

    def __init__(self, p: float = 0.3, rng=None):
        self.p = p
        self.rng = np.random.default_rng(1) if rng is None else rng

    def forward(self, x):
        if not self.training or self.p <= 0:
            return x
        self.mask = (self.rng.random(x.shape) >= self.p).astype(x.dtype) / (1.0 - self.p)
        return x * self.mask

    def backward(self, dout):
        if not self.training or self.p <= 0:
            return dout
        return dout * self.mask

    def parameters(self):
        return []

    def shapes(self):
        return {'type': 'dropout', 'p': self.p}


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class CharCNN:
    """conv -> relu -> pool -> conv -> relu -> pool -> dense -> relu -> dropout -> dense."""

    def __init__(self, in_size: int = 32, channels=(16, 32), hidden: int = 128,
                 n_classes: int = 7, dropout: float = 0.3, seed: int = 0):
        rng = np.random.default_rng(seed)
        c1, c2 = channels
        side = in_size // 4
        layers = [Conv2D(1, c1, 3, 1, rng, 'conv1'), ReLU(), MaxPool2x2(),
                  Conv2D(c1, c2, 3, 1, rng, 'conv2'), ReLU(), MaxPool2x2(),
                  Flatten(), Dense(c2 * side * side, hidden, rng, 'fc1'), ReLU(),
                  Dropout(dropout, rng), Dense(hidden, n_classes, rng, 'fc2')]
        self.layers = layers
        self.config = {'in_size': in_size, 'channels': list(channels), 'hidden': hidden,
                       'n_classes': n_classes, 'dropout': dropout, 'seed': seed}

    # -- forward / backward ------------------------------------------------- #
    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.training = training
            x = layer.forward(x)
        return x

    def backward(self, dlogits: np.ndarray) -> None:
        for layer in reversed(self.layers):
            dlogits = layer.backward(dlogits)

    def parameters(self):
        """List of (name, value, gradient, apply_weight_decay)."""
        params = []
        for layer in self.layers:
            for pname, value, grad, decay in layer.parameters():
                params.append((f'{layer.name}.{pname}', value, grad, decay))
        return params

    def n_parameters(self) -> int:
        return int(sum(v.size for _, v, _, _ in self.parameters()))

    # -- loss --------------------------------------------------------------- #
    @staticmethod
    def softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    def loss_only(self, x: np.ndarray, y: np.ndarray) -> float:
        """Cross-entropy without touching any gradient (used by the gradient check)."""
        probs = self.softmax(self.forward(x, training=False))
        return float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, None)).mean())

    def loss_and_grad(self, x: np.ndarray, y: np.ndarray, weight_decay: float = 0.0):
        """Softmax cross-entropy, its gradient, and the full backward pass.

        L2 regularisation is not added to the loss here: it is applied by the
        optimiser (Adam with decoupled weight decay), which keeps the reported
        loss equal to the data term.
        """
        logits = self.forward(x, training=True)
        probs = self.softmax(logits)
        n = x.shape[0]
        loss = -np.log(np.clip(probs[np.arange(n), y], 1e-12, None)).mean()
        dlogits = probs
        dlogits[np.arange(n), y] -= 1.0
        dlogits /= n
        self.backward(dlogits.astype(np.float32))
        return loss

    def predict_proba(self, x: np.ndarray, batch_size: int = 256) -> np.ndarray:
        out = []
        for start in range(0, x.shape[0], batch_size):
            out.append(self.softmax(self.forward(x[start:start + batch_size], training=False)))
        return np.concatenate(out, axis=0)

    def predict(self, x: np.ndarray, batch_size: int = 256) -> np.ndarray:
        return self.predict_proba(x, batch_size).argmax(axis=1)

    # -- persistence -------------------------------------------------------- #
    def state_dict(self) -> dict:
        return {name: value.copy() for name, value, _, _ in self.parameters()}

    def load_state_dict(self, state: dict) -> None:
        for name, value, _, _ in self.parameters():
            value[...] = state[name]

    def to_dtype(self, dtype) -> 'CharCNN':
        """Cast every parameter in place (the gradient check runs in float64)."""
        for layer in self.layers:
            for attr in ('W', 'b'):
                if hasattr(layer, attr):
                    setattr(layer, attr, getattr(layer, attr).astype(dtype))
                    setattr(layer, 'd' + attr, getattr(layer, 'd' + attr).astype(dtype))
        return self


class Adam:
    """Adam with optional L2 weight decay applied to the gradient."""

    def __init__(self, params, lr: float = 1e-3, weight_decay: float = 0.0,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.params = params
        self.lr, self.wd, self.beta1, self.beta2, self.eps = lr, weight_decay, beta1, beta2, eps
        self.m = {name: np.zeros_like(value) for name, value, _, _ in params}
        self.v = {name: np.zeros_like(value) for name, value, _, _ in params}
        self.t = 0

    def step(self) -> None:
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        for name, value, grad, decay in self.params:
            g = grad + self.wd * value if (self.wd and decay) else grad
            self.m[name] = b1 * self.m[name] + (1 - b1) * g
            self.v[name] = b2 * self.v[name] + (1 - b2) * (g * g)
            m_hat = self.m[name] / (1 - b1 ** self.t)
            v_hat = self.v[name] / (1 - b2 ** self.t)
            value -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# --------------------------------------------------------------------------- #
# Training utilities
# --------------------------------------------------------------------------- #
def augment_batch(batch: np.ndarray, rng, size: int, extra: int,
                  contrast: float = 0.15, brightness: float = 0.08) -> np.ndarray:
    """Random translation (crop from a padded copy) plus contrast/brightness jitter."""
    n = batch.shape[0]
    out = np.empty((n, 1, size, size), dtype=np.float32)
    for i in range(n):
        oy = rng.integers(0, 2 * extra + 1) if extra > 0 else 0
        ox = rng.integers(0, 2 * extra + 1) if extra > 0 else 0
        out[i, 0] = batch[i, 0, oy:oy + size, ox:ox + size]
    if contrast > 0:
        c = 1.0 + rng.uniform(-contrast, contrast, size=(n, 1, 1, 1)).astype(np.float32)
        b = rng.uniform(-brightness, brightness, size=(n, 1, 1, 1)).astype(np.float32)
        out = np.clip((out - 0.5) * c + 0.5 + b, 0.0, 1.0)
    return out


def fit(model: CharCNN, x_train: np.ndarray, y_train: np.ndarray,
        x_val: np.ndarray, y_val: np.ndarray, epochs: int = 15, batch_size: int = 64,
        lr: float = 1e-3, weight_decay: float = 1e-4, augment: bool = True,
        extra: int = 4, seed: int = 0, verbose: bool = True, log_every: int = 5) -> dict:
    """Mini-batch training with early stopping on the validation accuracy."""
    rng = np.random.default_rng(seed)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {'loss': [], 'train_accuracy': [], 'val_accuracy': [], 'val_loss': [], 'lr': lr}
    best = {'accuracy': -1.0, 'epoch': 0, 'state': model.state_dict()}
    n = x_train.shape[0]
    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)
        epoch_loss, seen, correct = 0.0, 0, 0
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            xb = x_train[idx]
            xb = augment_batch(xb, rng, model.config['in_size'], extra) if augment else xb
            yb = y_train[idx]
            loss = model.loss_and_grad(xb, yb, weight_decay)
            optimizer.step()
            epoch_loss += loss * len(idx)
            seen += len(idx)
            correct += int((model.forward(xb, training=False).argmax(axis=1) == yb).sum())
        val_probs = model.predict_proba(x_val, batch_size=256)
        val_acc = float((val_probs.argmax(axis=1) == y_val).mean())
        val_loss = float(-np.log(np.clip(val_probs[np.arange(len(y_val)), y_val], 1e-12, None)).mean())
        history['loss'].append(float(epoch_loss / seen))
        history['train_accuracy'].append(float(correct / seen))
        history['val_accuracy'].append(float(val_acc))
        history['val_loss'].append(float(val_loss))
        if val_acc > best['accuracy']:
            best = {'accuracy': val_acc, 'epoch': epoch, 'state': model.state_dict()}
        if verbose and (epoch % log_every == 0 or epoch == epochs or epoch == 1):
            print(f'    epoch {epoch:3d}/{epochs}: loss={epoch_loss / seen:.4f} '
                  f'train_acc={correct / seen:.4f} val_acc={val_acc:.4f} '
                  f'(best {best["accuracy"]:.4f} @ {best["epoch"]})', flush=True)
    model.load_state_dict(best['state'])
    history['best_epoch'] = best['epoch']
    history['best_val_accuracy'] = best['accuracy']
    return history


def numerical_gradient_check(seed: int = 0, eps: float = 1e-6) -> dict:
    """Finite-difference check of every analytic gradient (used as a unit test).

    The check runs in float64: in float32 the central difference of a loss of
    order one is dominated by rounding, which would hide a real mistake.
    """
    rng = np.random.default_rng(seed)
    model = CharCNN(in_size=8, channels=(2, 3), hidden=5, n_classes=4, dropout=0.0, seed=seed)
    model.to_dtype(np.float64)
    x = rng.random((3, 1, 8, 8))
    y = rng.integers(0, 4, size=3)
    model.loss_and_grad(x, y)
    report = {}
    for name, value, grad, _ in model.parameters():
        flat = value.ravel()
        analytic = grad.ravel().copy()
        picks = rng.choice(flat.size, size=min(6, flat.size), replace=False)
        worst = 0.0
        for p in picks:
            original = flat[p]
            flat[p] = original + eps
            plus = model.loss_only(x, y)
            flat[p] = original - eps
            minus = model.loss_only(x, y)
            flat[p] = original
            numeric = (plus - minus) / (2 * eps)
            denom = max(1e-6, abs(numeric), abs(analytic[p]))
            worst = max(worst, abs(numeric - analytic[p]) / denom)
        report[name] = float(worst)
    return report
