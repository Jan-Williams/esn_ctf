import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
import scipy.sparse
import scipy.stats
from jax.experimental import sparse
import typing
import numpy as np

jax.config.update("jax_enable_x64", True)

class LinearEmbedding(eqx.Module):
    """Linear embedding layer.

    Attributes
    ----------
    in_dim : int
        Reservoir input dimension.
    res_dim : int
        Reservoir dimension.
    scaling : float
        Min/max values of input matrix.
    win : Array
        Input matrix.
    chunks : int
        Number of parallel reservoirs.
    locality : int
        Adjacent reservoir overlap.

    Methods
    -------
    __call__(in_state)
        Embed input state to reservoir dimension.
    localize(in_state, periodic=True)
        Decompose input_state to parallel network inputs.
    moving_window(a)
        Helper function for localize.
    embed(in_state)
        Embed single input state to reservoir dimension.
    """

    in_dim: int
    res_dim: int
    scaling: float
    win: Array
    dtype: Float
    chunks: int
    locality: int
    group_size: int

    def __init__(
        self,
        in_dim: int,
        res_dim: int,
        scaling: float,
        dtype: Float = jnp.float64,
        chunks: int = 1,
        locality: int = 0,
        *,
        seed: int,
    ) -> None:
        """Instantiate linear embedding.

        Parameters
        ----------
        in_dim : int
            Input dimension to reservoir.
        res_dim : int
            Reservoir dimension.
        scaling : float
            Min/max values of input matrix.
        seed : int
            Random seed for generating the PRNG key for the reservoir computer.
        dtype : Float
            Dtype of model, jnp.float64 or jnp.float32.
        """
        self.res_dim = res_dim
        self.in_dim = in_dim
        self.dtype = dtype
        if not isinstance(res_dim, int):
            raise TypeError("Reservoir dimension res_dim must be an integer.")
        if not isinstance(in_dim, int):
            raise TypeError("Reservoir dimension res_dim must be an integer.")
        if not (dtype == jnp.float64 or dtype == jnp.float32):
            raise TypeError("dtype must be jnp.float64 of jnp.float32.")
        self.scaling = scaling
        key = jax.random.key(seed)
        self.group_size = int(in_dim / chunks)

        if in_dim % chunks:
            raise ValueError(
                f"The number of chunks {chunks} must evenly divide in_dim {in_dim}."
            )

        self.win = jax.random.uniform(
            key,
            (chunks, res_dim, self.group_size + 2 * locality),
            minval=-scaling,
            maxval=scaling,
            dtype=dtype,
        )
        self.locality = locality
        self.chunks = chunks

    @eqx.filter_jit
    def moving_window(self, a):
        """Generate window to compute localized states."""
        size = int(self.in_dim / self.chunks + 2 * self.locality)
        starts = jnp.arange(len(a) - size + 1)[: self.chunks] * int(
            self.in_dim / self.chunks
        )
        return eqx.filter_vmap(
            lambda start: jax.lax.dynamic_slice(a, (start,), (size,))
        )(starts)

    @eqx.filter_jit
    def localize(self, in_state: Array, periodic=True) -> Array:
        """Generate parallel reservoir inputs from input state.

        Parameters
        ----------
        in_state : Array
            Input state, (shape=(in_dim,))
        periodic : bool
            Assume periodic boundary conditions.

        Returns
        -------
        Array
            Parallel reservoir inputs, (shape=(chunks, group_size + 2*locality))
        """
        if len(in_state.shape) != 1:
            raise ValueError(
                "Only 1-dimensional localization is currently supported, detected a "
                f"{len(in_state.shape)}D field."
            )
        aug_state = jnp.hstack(
            [in_state[-self.locality :], in_state, in_state[: self.locality]]
        )
        if not periodic:
            aug_state = aug_state.at[: self.locality].set(aug_state[self.locality])
            aug_state = aug_state.at[-self.locality :].set(aug_state[-self.locality])
        return self.moving_window(aug_state)

    @eqx.filter_jit
    def embed(self, in_state: Array) -> Array:
        """Embed single state to reservoir dimensions.

        Parameters
        ----------
        in_state : Array
            Input state, (shape=(in_dim,)).

        Returns
        -------
        Array
            Embedded input to reservoir, (shape=(chunks, res_dim,)).
        """
        if in_state.shape != (self.in_dim,):
            print(in_state.shape)
            print(self.in_dim)
            raise ValueError("Incorrect dimension for input state.")
        localized_states = self.localize(in_state)

        return eqx.filter_vmap(jnp.matmul)(self.win, localized_states)

    def __call__(self, in_state: Array) -> Array:
        """Embed state to reservoir dimensions.

        Parameters
        ----------
        in_state : Array
            Input state, (shape=(in_dim,) or shape=(seq_len, in_dim)).

        Returns
        -------
        Array
            Embedded input to reservoir, (shape=(chunks, res_dim,) or
            shape=(seq_len, chunks, res_dim)).
        """
        if len(in_state.shape) == 1:
            to_ret = self.embed(in_state)
        elif len(in_state.shape) == 2:
            to_ret = self.batch_embed(in_state)
        else:
            raise ValueError(
                "Only 1-dimensional localization is currently supported, detected a "
                f"{len(in_state.shape) - 1}D field."
            )
        return to_ret

    @eqx.filter_jit
    def batch_embed(
        self,
        in_state: Array,
    ) -> Array:
        """Batch apply embedding from input states.

        Parameters
        ----------
        in_state : Array
            Input states.

        Returns
        -------
        Array
            Embedded input states to reservoir, (shape=(batch_dim, res_dim,)).
        """
        return eqx.filter_vmap(self.embed)(in_state)

class ESNDriver(eqx.Module):
    """Standard implementation of ESN reservoir with tanh nonlinearity.

    Attributes
    ----------
    res_dim : int
        Reservoir dimension.
    wr : Array
        Reservoir update matrix, (shape=(chunks, res_dim, res_dim,)).
    leak : float
        Leak rate parameter.
    spectral_radius : float
        Spectral radius of wr.
    density : float
        Density of wr.
    bias : float
        Additive bias in tanh nonlinearity.
    chunks: int
        Number of parallel reservoirs.
    dtype : Float
        Dtype, default jnp.float64.

    Methods
    -------
    advance(proj_vars, res_state)
        Updated reservoir state.
    __call__(proj_vars, res_state)
        Batched or single update to reservoir state.
    """

    res_dim: int
    leak: float
    spectral_radius: float
    density: float
    bias: float
    dtype: Float
    wr: Array
    chunks: int

    def __init__(
        self,
        res_dim: int,
        leak: float = 0.6,
        spectral_radius: float = 0.8,
        density: float = 0.02,
        bias: float = 1.6,
        dtype: Float = jnp.float64,
        chunks: int = 1,
        *,
        seed: int,
    ) -> None:
        """Initialize weight matrices.

        Parameters
        ----------
        res_dim : int
            Reservoir dimension.
        leak : float
            Leak rate parameter.
        spectral_radius : float
            Spectral radius of wr.
        density : float
            Density of wr.
        bias : float
            Additive bias in tanh nonlinearity.
        chunks: int
            Number of parallel reservoirs.
        dtype : Float
            Dtype, default jnp.float64.
        seed : int
            Random seed for generating the PRNG key for the reservoir computer.
        """
        self.res_dim = res_dim
        if not isinstance(res_dim, int):
            raise TypeError("Reservoir dimension res_dim must be an integer.")
        self.dtype = dtype
        if not (dtype == jnp.float64 or dtype == jnp.float32):
            raise TypeError("dtype must be jnp.float64 or jnp.float32.")
        self.leak = leak
        self.spectral_radius = spectral_radius
        self.density = density
        self.bias = bias
        self.dtype = dtype
        key = jax.random.key(seed)
        if spectral_radius <= 0:
            raise ValueError("Spectral radius must be positve.")
        if leak < 0 or leak > 1:
            raise ValueError("Leak rate must satisfy 0 < leak < 1.")
        if density < 0 or density > 1:
            raise ValueError("Density must satisfy 0 < density < 1.")
        wrkey1, wrkey2 = jax.random.split(key, 2)

        temp_list = []
        for jj in range(chunks):
            rng = np.random.default_rng(int(seed + jj))
            data_sampler = scipy.stats.uniform(loc=-1, scale=2).rvs
            sp_mat = scipy.sparse.random_array(
                (res_dim, res_dim), density=density, rng=rng, data_sampler=data_sampler
            )
            eigvals, _ = scipy.sparse.linalg.eigs(sp_mat, k=1)
            sp_mat = sp_mat * spectral_radius / np.abs(eigvals[0])
            jax_mat = jax.experimental.sparse.BCOO.from_scipy_sparse(sp_mat)
            jax_mat = jax.experimental.sparse.bcoo_broadcast_in_dim(
                jax_mat, shape=(1, res_dim, res_dim), broadcast_dimensions=(1, 2)
            )
            temp_list.append(jax_mat)
        wr = jax.experimental.sparse.bcoo_concatenate(temp_list, dimension=0)

        self.wr = wr
        self.chunks = chunks

    @eqx.filter_jit
    def advance(self, proj_vars: Array, res_state: Array) -> Array:
        """Advance the reservoir state.

        Parameters
        ----------
        proj_vars : Array
            Reservoir projected inputs, (shape=(chunks, res_dim,)).
        res_state : Array
            Reservoir state, (shape=(chunks, res_dim,)).

        Returns
        -------
        res_next : Array
            Reservoir state, (shape=(chunks, res_dim,)).
        """
        if proj_vars.shape != (self.chunks, self.res_dim):
            raise ValueError(f"Incorrect proj_var dimension, got {proj_vars.shape}")
        return (
            self.leak
            * self.sparse_ops(
                self.wr, res_state, proj_vars, self.bias * jnp.ones_like(proj_vars)
            )
            + (1 - self.leak) * res_state
        )

    @staticmethod
    @sparse.sparsify
    @jax.vmap
    def sparse_ops(wr, res_state, proj_vars, bias):
        """Dense operation to sparsify for advancing reservoir."""
        return jnp.tanh(wr @ res_state + proj_vars + bias)

    def __call__(self, proj_vars: Array, res_state: Array) -> Array:
        """Advance reservoir state.

        Parameters
        ----------
        proj_vars : Array
            Reservoir projected inputs, (shape=(chunks, res_dim) or
            shape=(seq_len, chunks, res_dim)).
        res_state : Array
            Current reservoir state, (shape=(chunks, res_dim) or
            shape=(seq_len, chunks, res_dim)).

        Returns
        -------
        Array
            Sequence of reservoir states, (shape=(chunks, res_dim,) or
            shape=(seq_len, chunks, res_dim)).
        """
        if len(proj_vars.shape) == 2:
            to_ret = self.advance(proj_vars, res_state)
        elif len(proj_vars.shape) == 3:
            to_ret = self.batch_advance(proj_vars, res_state)
        else:
            raise ValueError(
                "Only 1-dimensional localization is currently supported, detected a "
                f"{len(proj_vars.shape)}D field."
            )
        return to_ret

    @eqx.filter_jit
    def batch_advance(self, proj_vars: Array, res_state: Array) -> Array:
        """
        Batch advance the reservoir given projected inputs and current state.

        Parameters
        ----------
        proj_vars : Array
            Reservoir projected inputs.
        res_state : Array
            Reservoir state.

        Returns
        -------
        Array
            Updated reservoir state.
        """
        return eqx.filter_vmap(self.advance)(proj_vars, res_state)

class QuadraticReadout(eqx.Module):
    """Quadratic readout layer.

    Attributes
    ----------
    out_dim : int
        Dimension of reservoir output.
    res_dim : int
        Reservoir dimension.
    chunks : int
        Number of parallel reservoirs.
    wout : Array
        Output matrix.
    dtype : Float
            Dtype, default jnp.float64.

    Methods
    -------
    readout(res_state)
        Map from reservoir state to output state with quadratic nonlinearity.
    __call__(res_state)
        Map from reservoir state to output state with quadratic nonlinearity,
        handles batch and single outputs.
    """

    out_dim: int
    res_dim: int
    wout: Array
    chunks: int
    dtype: Float

    def __init__(
        self,
        out_dim: int,
        res_dim: int,
        chunks: int = 1,
        dtype: Float = jnp.float64,
        *,
        seed: int = 0,
    ) -> None:
        """Initialize readout layer to zeros.

        Parameters
        ----------
        out_dim : int
            Dimension of reservoir output.
        res_dim : int
            Reservoir dimension.
        chunks : int
            Number of parallel resrevoirs.
        dtype : Float
            Dtype, default jnp.float64.
        seed : int
            Not used for LinearReadout, present to maintain consistent interface.
        """
        self.res_dim = res_dim
        self.out_dim = out_dim
        self.dtype = dtype
        if not isinstance(res_dim, int):
            raise TypeError("Reservoir dimension res_dim must be an integer.")
        if not isinstance(out_dim, int):
            raise TypeError("Reservoir dimension res_dim must be an integer.")
        self.dtype = dtype
        if not (dtype == jnp.float64 or dtype == jnp.float32):
            raise TypeError("dtype must be jnp.float64 of jnp.float32.")
        self.wout = jnp.zeros((chunks, int(out_dim / chunks), res_dim), dtype=dtype)
        self.chunks = chunks

    @eqx.filter_jit
    def readout(self, res_state: Array) -> Array:
        """Readout from reservoir state.

        Parameters
        ----------
        res_state : Array
            Reservoir state, (shape=(chunks, res_dim,)).

        Returns
        -------
        Array
            Output from reservoir, (shape=(out_dim,)).
        """
        if res_state.shape[1] != self.res_dim:
            raise ValueError(
                "Incorrect reservoir dimension for instantiated output map."
            )
        res_state = res_state.at[:, ::2].set(res_state[:, ::2] * res_state[:, ::2])
        return jnp.ravel(eqx.filter_vmap(jnp.matmul)(self.wout, res_state))

    def __call__(self, res_state: Array) -> Array:
        """Call either readout or batch_readout depending on dimensions.

        Parameters
        ----------
        res_state : Array
            Reservoir state, (shape=(chunks, res_dim) or
            shape=(seq_len, chunks, res_dim)).

        Returns
        -------
        Array
            Output state, (out_dim,) or shape=(seq_len, out_dim)).
        """
        if len(res_state.shape) == 2:
            to_ret = self.readout(res_state)
        elif len(res_state.shape) == 3:
            to_ret = self.batch_readout(res_state)
        else:
            raise ValueError(
                "Only 1-dimensional localization is currently supported, detected a "
                f"{len(res_state.shape)}D field."
            )
        return to_ret

    @eqx.filter_jit
    def batch_readout(
        self,
        res_state: Array,
    ) -> Array:
        """Batch apply readout from reservoir states.

        Parameters
        ----------
        res_state : Array
            Reservoir state.

        Returns
        -------
        Array
            Output from reservoir states.
        """
        return eqx.filter_vmap(self.readout)(res_state)

class ESN(eqx.Module):
    """
    Basic implementation of ESN for forecasting.

    Attributes
    ----------
    res_dim : int
        Reservoir dimension.
    data_dim : int
        Input/output dimension.
    driver : ESNDriver
        Driver implmenting the Echo State Network dynamics.
    readout : QuadraticReadout
        Trainable linear readout layer.
    embedding : LinearEmbedding
        Untrainable linear embedding layer.

    Methods
    -------
    force(in_seq, res_state)
        Teacher forces the reservoir with sequence in_seq and init. cond. res_state.
    forecast(fcast_len, res_state)
        Perform a forecast of fcast_len steps from res_state.
    set_readout(readout)
        Replace readout layer.
    set_embedding(embedding)
        Replace embedding layer.
    """

    driver: typing.Any
    readout: typing.Any
    embedding: typing.Any
    data_dim: int
    in_dim: int
    out_dim: int
    res_dim: int
    dtype: Float
    seed: int

    def __init__(
        self,
        data_dim: int,
        res_dim: int,
        leak_rate: float = 0.6,
        bias: float = 1.6,
        embedding_scaling: float = 0.08,
        Wr_density: float = 0.02,
        Wr_spectral_radius: float = 0.8,
        dtype: type = jnp.float64,
        seed: int = 0,
        chunks: int = 1,
        locality: int = 0,
    ) -> None:
        """
        Initialize the ESN model.

        Parameters
        ----------
        data_dim : int
            Dimension of the input data.
        res_dim : int
            Dimension of the reservoir adjacency matrix Wr.
        leak_rate : float
            Integration leak rate of the reservoir dynamics.
        bias : float
            Bias term for the reservoir dynamics.
        embedding_scaling : float
            Scaling factor for the embedding layer.
        Wr_density : float
            Density of the reservoir adjacency matrix Wr.
        Wr_spectral_radius : float
            Largest eigenvalue of the reservoir adjacency matrix Wr.
        dtype : type
            Data type of the model (jnp.float64 is highly recommended).
        seed : int
            Random seed for generating the PRNG key for the reservoir computer.
        chunks : int
            Number of parallel reservoirs, must evenly divide data_dim.
        locality : int
            Overlap in adjacent parallel reservoirs.
        quadratic : bool
            Use quadratic nonlinearity in output, default False.
        """
        # Initialize the random key and reservoir dimension
        key = jax.random.PRNGKey(seed)
        key_driver, key_readout, key_embedding = jax.random.split(key, 3)
        np.random.seed(seed=seed)
        # init in embedding, driver and readout
        embedding = LinearEmbedding(
            in_dim=data_dim,
            res_dim=res_dim,
            seed=key_embedding[0],
            scaling=embedding_scaling,
            chunks=chunks,
            locality=locality,
        )
        driver = ESNDriver(
            res_dim=res_dim,
            seed=key_driver[0],
            leak=leak_rate,
            bias=bias,
            density=Wr_density,
            spectral_radius=Wr_spectral_radius,
            chunks=chunks,
        )

        readout = QuadraticReadout(
                out_dim=data_dim, res_dim=res_dim, seed=key_readout[0], chunks=chunks
        )

        self.driver = driver
        self.readout = readout
        self.embedding = embedding
        self.data_dim = data_dim
        self.res_dim = res_dim
        self.in_dim = data_dim
        self.out_dim = data_dim
        self.dtype = dtype
        self.seed = seed


    @eqx.filter_jit
    def forecast(self, fcast_len: int, res_state: Array) -> Array:
        """Forecast from an initial reservoir state.

        Parameters
        ----------
        fcast_len : int
            Steps to forecast.
        res_state : Array
            Initial reservoir state, (shape=(res_dim)).

        Returns
        -------
        Array
            Forecasted states, (shape=(fcast_len, data_dim))
        """

        def scan_fn(state, _):
            out_state = self.driver(self.embedding(self.readout(state)), state)
            return (out_state, self.readout(out_state))

        _, state_seq = jax.lax.scan(scan_fn, res_state, None, length=fcast_len)
        return state_seq

    @eqx.filter_jit
    def force(self, in_seq: Array, res_state: Array) -> Array:
        """Teacher forces the reservoir.

        Parameters
        ----------
        in_seq: Array
            Input sequence to force the reservoir, (shape=(seq_len, data_dim)).
        res_state : Array
            Initial reservoir stat, (shape=(chunks, res_dim,)).

        Returns
        -------
        Array
            Forced reservoir sequence, (shape=(seq_len, res_dim)).
        """
        def scan_fn(state, in_vars):
            proj_vars = self.embedding(in_vars)
            res_state = self.driver(proj_vars, state)
            return (res_state, res_state)

        _, res_seq = jax.lax.scan(scan_fn, res_state, in_seq)
        return res_seq

    def set_readout(self, readout):
        """Replace readout layer.

        Parameters
        ----------
        readout : QuadraticReadout
            New readout layer.

        Returns
        -------
        ESN
            Updated model with new readout layer.
        """

        def where(m):
            return m.readout

        new_model = eqx.tree_at(where, self, readout)
        return new_model

    def set_embedding(self, embedding):
        """Replace embedding layer.

        Parameters
        ----------
        embedding : LinearEmbedding
            New embedding layer.

        Returns
        -------
        ESN
            Updated model with new embedding layer.
        """

        def where(m):
            return m.embedding

        new_model = eqx.tree_at(where, self, embedding)
        return new_model

    @eqx.filter_jit
    def denoise(self, in_seq: Array, res_state: Array):
        """Use teacher forced reservoir to denoise in_seq.

        Parameters
        ----------
        in_seq: Array
            Input sequence to force the reservoir, (shape=(seq_len, data_dim)).
        res_state : Array
            Initial reservoir stat, (shape=(chunks, res_dim,)).

        Returns
        -------
        Array
            Forced reservoir sequence, (shape=(seq_len, res_dim)).
        """
        res_seq = self.force(in_seq, res_state)
        denoised = self.readout(res_seq)
        return denoised


def train_ESN_forecaster(
    model: ESN,
    train_seq: Array,
    target_seq: Array = None,
    spinup: int = 0,
    initial_res_state: Array = None,
    beta: float = 8e-8,
) -> tuple[ESN, Array]:
    """Training function for RC forecaster.

    Parameters
    ----------
    model : ESN
        ESN model to train.
    train_seq : Array
        Training input sequence for reservoir, (shape=(seq_len, data_dim)).
    target_seq : Array
        Target sequence for training reservoir, (shape=(seq_len, data_dim)).
    initial_res_state : Array
        Initial reservoir state, (shape=(chunks, res_dim,)).
    spinup : int
        Initial transient of reservoir states to discard.
    beta : float
        Tikhonov regularization parameter.

    Returns
    -------
    model : ESN
        Trained ESN model.
    res_seq : Array
        Training sequence of reservoir states.
    """
    if initial_res_state is None:
        initial_res_state = jnp.zeros(
            (
                model.embedding.chunks,
                model.res_dim,
            ),
            dtype=model.dtype,
        )

    if target_seq is None:
        target_seq = train_seq[:, 1:, :]
        train_seq = train_seq[:,:-1, :]

    res_seqs = jnp.empty((target_seq.shape[0], target_seq.shape[1] - spinup, model.embedding.chunks, model.driver.res_dim))
    target_seqs = target_seq[:, spinup:, :]
    for jj in range(target_seq.shape[0]):
        res_seq = model.force(train_seq[jj], initial_res_state)
        res_seqs = res_seqs.at[jj].set(res_seq[spinup:])

    res_seq_train = res_seqs.at[:, :, :, ::2].set(res_seqs[:, :, :, ::2] ** 2)
    res_seq_train = jnp.concatenate(res_seq_train, axis=0)
    target_seq = jnp.concatenate(target_seqs, axis=0)



    def solve_single_ridge_reg(res_seq, target_seq, beta):
        lhs = res_seq.T @ res_seq + beta * jnp.eye(
            res_seq.shape[1], dtype=res_seq.dtype
        )
        rhs = res_seq.T @ target_seq
        cmat = jax.scipy.linalg.solve(lhs, rhs, assume_a="sym").T
        return cmat

    solve_all_ridge_reg = eqx.filter_vmap(
        solve_single_ridge_reg, in_axes=eqx.if_array(1)
    )
    cmat = solve_all_ridge_reg(
        res_seq_train,
        target_seq.reshape(res_seq_train.shape[0], res_seq_train.shape[1], -1),
        beta,
    )

    def where(m):
        return m.readout.wout
    model = eqx.tree_at(where, model, cmat)

    return model, res_seq
