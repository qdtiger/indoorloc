"""Neural-network localizers behind the BaseLocalizer contract: DeepLocalizer, MLPLocalizer, CNN1DLocalizer.

This module imports numpy only: a deep localizer can be created, cloned, listed and grid-searched
without torch. torch (extra ``[deep]``) is imported when the model is fitted, used or loaded, from
the bridge modules ``backbones``, ``heads`` and ``training`` of this package.
"""
from __future__ import annotations

import numpy as np

from ...core import Prediction, SampleTable, requires
from ..base import BaseLocalizer

__all__ = ["CNN1DLocalizer", "DeepLocalizer", "MLPLocalizer"]


def _training():
    requires("torch", "deep")  # a clear "pip install 'indoorloc[deep]'" when torch is missing
    from . import training

    return training


class DeepLocalizer(BaseLocalizer):
    """Deep fingerprinting: a backbone network with multi-task heads, trained end to end.

    A backbone (``"mlp"``, ``"cnn1d"`` or any timm model name) turns a scan into features; a
    multi-task head regresses the coordinates and, when the training data has them, classifies the
    floor and the building (Kim et al., 2018). Targets are standardized with float64 statistics
    stored in the model (mean per axis and one common scale, so the position loss stays proportional
    to the squared Euclidean error); inputs are standardized with one global mean and scale
    (``scale_inputs``), which keeps dBm and [0, 1] features equally usable and does not blow up APs
    that are rarely heard. Training is mini-batch AdamW on ``MSE(pos) + CE(floor) + CE(building)``
    (weighted by ``task_weights``) with early stopping on a validation set, and is deterministic for
    a given ``random_state``, input and torch thread count (CPU by default). Training is chaotic
    in the last bit, though: on UJIIndoorLoc a relative 4e-17 change of the position mean moves
    the mean error by 0.4 m, and so do other seeds, so compare deep models over several seeds. ``X`` must be complete: missing
    readings (NaN) are refused with a pointer to ``preprocess=FillMissing(...)``.

    Parameters
    ----------
    backbone : str, default "mlp"
        ``"mlp"``, ``"cnn1d"`` or a timm model name (``"resnet18"``, ``"efficientnet_b0"``, ...; needs timm).
        ``"mlp"`` and ``"cnn1d"`` start from the defaults of ``MLPLocalizer`` and ``CNN1DLocalizer``, so
        ``DeepLocalizer(backbone="cnn1d")`` builds the same network as ``CNN1DLocalizer()``.
    hidden : tuple of int or None, default None
        Hidden widths of the MLP, or the channels of the CNN1D; unused by timm backbones.
        None = the backbone's default ((512, 256, 128) for the MLP, (32, 64, 128) for the CNN1D).
    dropout : float or None, default None
        Dropout of the backbone layers (timm: ``drop_rate`` before the head) and of the head's hidden
        layers. None = the backbone's default (0.3 for the MLP, 0 for the CNN1D and timm).
    batch_norm : bool, default True
        BatchNorm after every MLP/CNN1D layer (then ``batch_size`` must be at least 2).
    head_hidden : tuple of int, default ()
        Hidden widths of each task branch; () = linear heads on the shared features.
    backbone_options : dict or None
        Extra backbone arguments, overriding the defaults: ``activation`` (MLP); ``kernel_sizes``,
        ``strides``, ``pooling``, ``output_length``, ``projection`` (CNN1D); ``image_size`` and any
        ``timm.create_model`` keyword (timm).
    pretrained : bool, default False
        timm backbones only: start from timm's pretrained weights (downloads them).
    epochs : int, default 100
        Maximum number of passes over the training data.
    batch_size : int, default 256
    lr, weight_decay : float, default 1e-3, 1e-4
        AdamW learning rate and (decoupled) weight decay.
    patience : int or None, default 10
        Early stopping: stop after this many epochs without a lower validation error and keep the
        best epoch's weights. The validation set is ``fit(..., eval_set=[(X_val, y_val)])`` (the last
        pair is used; ``X_val`` may be a SampleTable, with ``y_val=None`` taking its ``pos``) or,
        without it, a random ``validation_fraction`` of the training data held out with
        ``random_state``. None trains all ``epochs`` on all data (an ``eval_set`` is then only
        recorded in ``history_``). The random hold-out is not grouped by reference point, so its error
        is optimistic; pass a held-out survey (e.g. another campaign or device) as ``eval_set``.
    validation_fraction : float, default 0.1
    task_weights : (float, float, float), default (1.0, 1.0, 1.0)
        Loss weights of position, floor and building.
    scale_inputs : bool, default True
        Standardize X with one global mean and standard deviation learned in ``fit``.
    device : str, default "cpu"
        torch device used for training and inference.
    random_state : int or None, default 0
        Seeds initialisation, dropout, shuffling and the validation split. None = not reproducible.
    verbose : int, default 0
        1 prints one line per epoch.

    Attributes
    ----------
    net_ : the trained ``LocalizationNet`` (torch module, evaluation mode).
    net_config_ : dict describing its architecture (JSON; used to rebuild it on ``load_model``).
    pos_mean_ (D,), pos_scale_ : target standardization (float64).  x_mean_, x_scale_ : input scaling.
    floor_classes_, building_classes_ : int64 labels of the classification heads, or None.
    history_ : ``{"loss": (n_epochs,), "val_error": (n_epochs,)}``, validation error in coordinate units.
    n_epochs_, best_epoch_ : epochs run and the 0-based epoch whose weights were kept (-1: the last).

    Saving: ``save`` stores the ``state_dict`` as numpy arrays (no pickle); ``load_model`` rebuilds
    the network from ``net_config_`` (never re-downloading pretrained weights) and loads them.

    Fixes relative to 0.1 ``DeepLocalizer``: targets are standardized (0.1 regressed raw
    coordinates such as UJIIndoorLoc's 4.86e6 m northings), models can be saved and loaded, early
    stopping uses a held-out validation set, and training is seeded without touching global RNGs.

    Deviations from the cited papers: Kim et al. pre-train a stacked autoencoder and treat building,
    floor and location as one multi-label *classification* (the position is then a weighted centroid
    of the predicted reference points); here the network is trained end to end and *regresses* the
    coordinates, with separate classification heads for floor and building. CNNLoc (Song et al.)
    likewise pre-trains an autoencoder and trains one network per task.

    References
    ----------
    K. S. Kim, S. Lee and K. Huang, "A scalable deep neural network architecture for multi-building
    and multi-floor indoor localization based on Wi-Fi fingerprinting", Big Data Analytics 3:4, 2018.
    https://doi.org/10.1186/s41044-018-0031-2

    X. Song, X. Fan, C. Xiang, Q. Ye, L. Liu, Z. Wang, X. He, N. Yang and G. Fang, "A Novel
    Convolutional Neural Network Based Indoor Localization Framework With WiFi Fingerprinting",
    IEEE Access 7:110698-110709, 2019. https://doi.org/10.1109/ACCESS.2019.2933921

    I. Loshchilov and F. Hutter, "Decoupled Weight Decay Regularization", ICLR, 2019.
    https://openreview.net/forum?id=Bkg6RiCqY7
    """

    _allow_nan = False

    def __init__(self, backbone: str = "mlp", hidden=None, dropout: float | None = None,
                 batch_norm: bool = True, head_hidden=(), backbone_options: dict | None = None,
                 pretrained: bool = False, epochs: int = 100, batch_size: int = 256, lr: float = 1e-3,
                 weight_decay: float = 1e-4, patience: int | None = 10, validation_fraction: float = 0.1,
                 task_weights=(1.0, 1.0, 1.0), scale_inputs: bool = True, device: str = "cpu",
                 random_state: int | None = 0, verbose: int = 0):
        self.backbone = backbone
        self.hidden = hidden
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.head_hidden = head_hidden
        self.backbone_options = backbone_options
        self.pretrained = pretrained
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.task_weights = task_weights
        self.scale_inputs = scale_inputs
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    # ------------------------------------------------------------------ configuration
    def _backbone_spec(self) -> tuple[str, dict]:
        """``(backbone name, keyword arguments)``; the subclasses fix the name.

        ``"mlp"``/``"cnn1d"`` start from ``MLPLocalizer()``/``CNN1DLocalizer()`` defaults; ``hidden``,
        ``dropout`` (when not None), ``batch_norm`` and then ``backbone_options`` override them."""
        options = dict(self.backbone_options or {})
        name = str(self.backbone)
        dedicated = {"mlp": MLPLocalizer, "cnn1d": CNN1DLocalizer}.get(name.lower())
        if dedicated is None:  # a timm model name
            return name, {**({} if self.dropout is None else {"drop_rate": float(self.dropout)}), **options}
        name, spec = dedicated()._backbone_spec()
        if self.hidden is not None:
            spec["hidden" if name == "mlp" else "channels"] = tuple(int(h) for h in self.hidden)
        if self.dropout is not None:
            spec["dropout"] = float(self.dropout)
        return name, {**spec, "batch_norm": bool(self.batch_norm), **options}

    def _check_params(self, backbone_kwargs: dict) -> None:
        dropout = backbone_kwargs.get("dropout", backbone_kwargs.get("drop_rate", 0.0))
        widths = backbone_kwargs.get("hidden", backbone_kwargs.get("channels", ()))
        batch_norm = bool(backbone_kwargs.get("batch_norm", False))
        checks = [
            (isinstance(self.epochs, (int, np.integer)) and self.epochs >= 1, "epochs must be an int >= 1"),
            (isinstance(self.batch_size, (int, np.integer)) and self.batch_size >= 1, "batch_size must be an int >= 1"),
            (not batch_norm or self.batch_size >= 2,
             "batch_size must be >= 2 with batch_norm=True (BatchNorm cannot normalise a single sample)"),
            (self.lr > 0, "lr must be > 0"),
            (self.weight_decay >= 0, "weight_decay must be >= 0"),
            (self.patience is None or (isinstance(self.patience, (int, np.integer)) and self.patience >= 1),
             "patience must be None or an int >= 1"),
            (0.0 <= self.validation_fraction < 1.0, "validation_fraction must be in [0, 1)"),
            (len(self.task_weights) == 3 and all(w >= 0 for w in self.task_weights),
             "task_weights must be three non-negative numbers (position, floor, building)"),
            (0.0 <= dropout < 1.0, "dropout must be in [0, 1)"),
            (all(int(w) >= 1 for w in widths), f"layer widths must be >= 1, got {tuple(widths)}"),
        ]
        for ok, message in checks:
            if not ok:
                raise ValueError(message)

    # ------------------------------------------------------------------ data handling
    def _inputs(self, X: np.ndarray) -> np.ndarray:
        """Scaled float32 network input (computed in float64)."""
        return ((np.asarray(X, dtype=np.float64) - self.x_mean_) / self.x_scale_).astype(np.float32)

    def _eval_arrays(self, eval_set, X: np.ndarray, n_outputs: int):
        """The last ``(X_val, y_val)`` pair of ``eval_set``, validated like the training data."""
        try:
            X_val, y_val = list(eval_set)[-1]
        except (TypeError, ValueError, IndexError):
            raise ValueError("eval_set must be a list of (X_val, y_val) pairs, "
                             "e.g. eval_set=[(X_val, y_val)]") from None
        if isinstance(X_val, SampleTable):  # (table, None): the table's own positions
            X_val, y_val = X_val.X, X_val.pos if y_val is None else y_val
        if y_val is None:
            raise ValueError("eval_set y is None: pass (X_val, y_val) or (SampleTable, None)")
        X_val = np.asarray(X_val)
        if X_val.dtype.kind not in "biuf" or X_val.shape[1:] != X.shape[1:] or len(X_val) == 0:
            raise ValueError(f"eval_set X must be real, non-empty and shaped (n, {', '.join(map(str, X.shape[1:]))}), "
                             f"got {X_val.dtype} {X_val.shape}")
        if not np.all(np.isfinite(X_val)):
            raise ValueError("eval_set X contains NaN or inf (missing readings?); fill them like X, e.g. "
                             "create_model(..., preprocess=FillMissing(-104)), which also preprocesses eval_set")
        y_val = np.asarray(y_val, dtype=np.float64)
        y_val = y_val[:, None] if y_val.ndim == 1 else y_val
        if y_val.shape != (len(X_val), n_outputs) or not np.all(np.isfinite(y_val)):
            raise ValueError(f"eval_set y must be finite and shaped ({len(X_val)}, {n_outputs}), got {y_val.shape}")
        return X_val, y_val

    # ------------------------------------------------------------------ BaseLocalizer hooks
    def _fit(self, X, pos, floor, building, eval_set=None):
        name, backbone_kwargs = self._backbone_spec()
        self._check_params(backbone_kwargs)
        training = _training()
        n, n_outputs = len(X), pos.shape[1]
        if self.scale_inputs:
            x_mean, x_scale = float(np.mean(X, dtype=np.float64)), float(np.std(X, dtype=np.float64))
            x_scale = x_scale if np.isfinite(x_scale) and x_scale > 0 else 1.0
        else:
            x_mean, x_scale = 0.0, 1.0
        pos_mean = pos.mean(0)
        pos_scale = float(np.sqrt(pos.var(0).mean()))
        pos_scale = pos_scale if pos_scale > 0 else 1.0
        labels = {}
        for key, values in (("floor", floor), ("building", building)):
            labels[key] = None if values is None else np.unique(values, return_inverse=True)
        validation = None if eval_set is None else self._eval_arrays(eval_set, X, n_outputs)

        split_seq, shuffle_seq, torch_seq = np.random.SeedSequence(self.random_state).spawn(3)
        train_rows = np.arange(n)
        if self.patience is not None and validation is None:
            n_val = int(np.ceil(self.validation_fraction * n))
            if n_val == 0:
                raise ValueError("early stopping (patience) needs eval_set=[(X_val, y_val)] or "
                                 "validation_fraction > 0; or pass patience=None")
            order = np.random.default_rng(split_seq).permutation(n)
            train_rows, val_rows = np.sort(order[n_val:]), np.sort(order[:n_val])
            validation = (X[val_rows], pos[val_rows])
        if len(train_rows) < 2:
            raise ValueError(f"{type(self).__name__} needs at least 2 training samples, got {len(train_rows)}")

        self.x_mean_, self.x_scale_ = x_mean, x_scale  # _inputs reads them
        head_dropout = float(backbone_kwargs.get("dropout", backbone_kwargs.get("drop_rate", 0.0)))
        config = {"backbone": name, "backbone_kwargs": backbone_kwargs,
                  "input_shape": tuple(int(s) for s in X.shape[1:]),
                  "n_outputs": int(n_outputs), "n_floors": 0 if labels["floor"] is None else len(labels["floor"][0]),
                  "n_buildings": 0 if labels["building"] is None else len(labels["building"][0]),
                  "head_hidden": tuple(int(h) for h in self.head_hidden), "head_dropout": head_dropout}
        scale = lambda P: (P - pos_mean) / pos_scale  # noqa: E731
        with training.seeded(int(torch_seq.generate_state(1)[0]), self.device):
            net = training.build_network(config, pretrained=self.pretrained)
            result = training.train_model(
                net, self._inputs(X[train_rows]), scale(pos[train_rows]),
                *(None if labels[k] is None else labels[k][1][train_rows] for k in ("floor", "building")),
                validation=None if validation is None else (self._inputs(validation[0]), scale(validation[1])),
                epochs=int(self.epochs), batch_size=int(self.batch_size), lr=float(self.lr),
                weight_decay=float(self.weight_decay), patience=self.patience, task_weights=tuple(self.task_weights),
                rng=np.random.default_rng(shuffle_seq), device=self.device, distance_scale=pos_scale,
                verbose=self.verbose)
        self.net_, self.net_config_ = net, config
        self.pos_mean_, self.pos_scale_ = pos_mean, pos_scale
        self.floor_classes_ = None if labels["floor"] is None else labels["floor"][0]
        self.building_classes_ = None if labels["building"] is None else labels["building"][0]
        self.history_ = {k: np.asarray(v, dtype=np.float64) for k, v in result.items() if k != "best_epoch"}
        self.n_epochs_ = len(result["loss"])
        self.best_epoch_ = int(result["best_epoch"]) if self.patience is not None else -1

    def _localize(self, X) -> Prediction:
        out = _training().predict_outputs(self.net_, self._inputs(X), device=self.device)
        pos = out["pos"].astype(np.float64) * self.pos_scale_ + self.pos_mean_
        vote = lambda classes, key: None if classes is None else classes[np.argmax(out[key], axis=1)]  # noqa: E731
        return Prediction(pos, vote(self.floor_classes_, "floor"), vote(self.building_classes_, "building"))

    # ------------------------------------------------------------------ persistence (no pickle)
    def _get_state(self) -> dict:
        state = super()._get_state()
        net = state.pop("net_", None)
        if net is not None:  # the module becomes its state_dict, as numpy arrays
            state["net_weights_"] = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
        return state

    def _set_state(self, state: dict) -> None:
        state = dict(state)
        weights = state.pop("net_weights_", None)
        super()._set_state(state)
        if weights is not None:
            training = _training()
            torch = requires("torch", "deep")
            with training.seeded(0):  # the initial weights are overwritten; the caller's RNG stays untouched
                net = training.build_network(self.net_config_, pretrained=False)
            net.load_state_dict({k: torch.as_tensor(np.array(v)) for k, v in weights.items()})
            self.net_ = net.to(self.device).eval()


class MLPLocalizer(DeepLocalizer):
    """Multi-layer perceptron fingerprinting: ``DeepLocalizer`` with the ``"mlp"`` backbone.

    ``hidden`` widths of ``Linear -> BatchNorm -> activation -> Dropout`` layers (the 0.1
    ``MLPBackbone`` defaults), followed by the multi-task head. The other parameters are those of
    ``DeepLocalizer``. Unlike Kim et al., who pre-train a stacked autoencoder and classify
    building, floor and reference point jointly, the network is trained end to end and regresses
    the coordinates (see ``DeepLocalizer``).

    References
    ----------
    K. S. Kim, S. Lee and K. Huang, "A scalable deep neural network architecture for multi-building
    and multi-floor indoor localization based on Wi-Fi fingerprinting", Big Data Analytics 3:4, 2018.
    https://doi.org/10.1186/s41044-018-0031-2
    """

    pretrained = False  # fixed by the architecture: a class attribute, not a parameter

    def __init__(self, hidden=(512, 256, 128), activation: str = "relu", dropout: float = 0.3,
                 batch_norm: bool = True, head_hidden=(), epochs: int = 100, batch_size: int = 256,
                 lr: float = 1e-3, weight_decay: float = 1e-4, patience: int | None = 10,
                 validation_fraction: float = 0.1, task_weights=(1.0, 1.0, 1.0), scale_inputs: bool = True,
                 device: str = "cpu", random_state: int | None = 0, verbose: int = 0):
        self.hidden = hidden
        self.activation = activation
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.head_hidden = head_hidden
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.task_weights = task_weights
        self.scale_inputs = scale_inputs
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    def _backbone_spec(self) -> tuple[str, dict]:
        return "mlp", {"hidden": tuple(int(h) for h in self.hidden), "activation": str(self.activation),
                       "dropout": float(self.dropout), "batch_norm": bool(self.batch_norm)}


class CNN1DLocalizer(DeepLocalizer):
    """1-D convolutional fingerprinting: ``DeepLocalizer`` with the ``"cnn1d"`` backbone.

    With the default ``projection=256`` the scan is first encoded by a dense layer (CNNLoc encodes it
    with a stacked autoencoder pre-trained separately; here one dense layer is trained end to end with
    the rest of the network), then convolved with ``channels``, ``kernel_sizes`` and ``strides``
    per layer, pooled (``pooling``) to ``output_length`` positions and passed to the multi-task head.
    ``projection=None`` convolves the raw feature axis: input ``(N, F)`` is one channel and
    ``(N, C, F)`` has C channels (use it when that axis is physical, e.g. CSI subcarriers). The other
    parameters are those of ``DeepLocalizer``.

    Defaults differ from 0.1 (channels 64-128-256, dropout 0.3 after every convolution, global pooling,
    no projection), which on UJIIndoorLoc gave a 48.7 m mean error against 11.3 m for the MLP. The
    new defaults were chosen after comparing variants on UJIIndoorLoc, partly on its validation split,
    so numbers reported on that split for them are optimistic.

    References
    ----------
    X. Song, X. Fan, C. Xiang, Q. Ye, L. Liu, Z. Wang, X. He, N. Yang and G. Fang, "A Novel
    Convolutional Neural Network Based Indoor Localization Framework With WiFi Fingerprinting",
    IEEE Access 7:110698-110709, 2019. https://doi.org/10.1109/ACCESS.2019.2933921
    """

    pretrained = False  # fixed by the architecture: a class attribute, not a parameter

    def __init__(self, channels=(32, 64, 128), kernel_sizes=(7, 5, 3), strides=(2, 2, 2), pooling: str = "max",
                 output_length: int = 8, projection: int | None = 256, dropout: float = 0.0,
                 batch_norm: bool = True, head_hidden=(), epochs: int = 100,
                 batch_size: int = 256, lr: float = 1e-3, weight_decay: float = 1e-4, patience: int | None = 10,
                 validation_fraction: float = 0.1, task_weights=(1.0, 1.0, 1.0), scale_inputs: bool = True,
                 device: str = "cpu", random_state: int | None = 0, verbose: int = 0):
        self.channels = channels
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.pooling = pooling
        self.output_length = output_length
        self.projection = projection
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.head_hidden = head_hidden
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.task_weights = task_weights
        self.scale_inputs = scale_inputs
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    def _backbone_spec(self) -> tuple[str, dict]:
        return "cnn1d", {"channels": tuple(int(c) for c in self.channels),
                         "kernel_sizes": tuple(int(k) for k in self.kernel_sizes),
                         "strides": tuple(int(s) for s in self.strides), "pooling": str(self.pooling),
                         "output_length": int(self.output_length),
                         "projection": None if self.projection is None else int(self.projection),
                         "dropout": float(self.dropout), "batch_norm": bool(self.batch_norm)}
