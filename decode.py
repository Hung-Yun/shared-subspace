#%% Imports

from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import decomposition
import data_loader as io
from tqdm.auto import tqdm

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

import warnings
warnings.filterwarnings('ignore')

from utils import *

plt.figure()
plt.close()
fig_set(font_size=10, linewidth=0.8)

class DecoderCV:
    """Run cross-validated binary decoding and store fold-level metrics.

    Parameters
    ----------
    model : sklearn estimator, default=LogisticRegression(solver='liblinear')
        Binary classifier used inside each training fold. The estimator must
        implement either `predict_proba` or `decision_function`.
    cv : sklearn CV splitter or None, default=None
        Cross-validation splitter with a `.split(X, y)` method. If `None`, a
        `StratifiedKFold` splitter is constructed from `cv_splits` and
        `random_state`.
    cv_splits : int, default=5
        Number of folds used only when `cv is None`.
    n_permutations : int, default=0
        Number of label permutations used to build a null distribution of mean
        test ROC AUC values.
    random_state : int, default=0
        Seed used for the default CV splitter and permutation testing.
    verbose : bool, default=VERBOSE
        If `True`, print fold and permutation progress.

    Attributes
    ----------
    train_roc_auc : list[float]
        ROC AUC values computed on the training partition of each fold.
    test_roc_auc : list[float]
        ROC AUC values computed on the held-out partition of each fold.
    test_pr_auc : list[float]
        Average precision / PR AUC values computed on the held-out partition of
        each fold.
    test_balanced_accuracy : list[float]
        Balanced accuracy values computed on the held-out partition of each
        fold using hard class predictions.
    permuted_auc : list[float]
        Mean held-out ROC AUC from each permutation repeat.
    p_value : float
        One-tailed permutation p-value comparing the observed mean test ROC AUC
        against `permuted_auc`. `np.nan` if no permutations were run.
    class_balance : float
        Fraction of samples with positive label (`y == 1`) in the most recent
        `fit` call.
    model_name : str
        Class name of the estimator passed in as `model`.
    """

    def __init__(
        self,
        model=LogisticRegression(solver='liblinear'),
        cv=None,
        cv_splits=5,
        n_permutations=0,
        random_state=0,
        verbose=VERBOSE,
    ):
        self.model = model
        self.cv = cv
        self.cv_splits = cv_splits
        self.n_permutations = n_permutations
        self.random_state = random_state
        self.verbose = verbose

        self.train_roc_auc = []
        self.test_roc_auc = []
        self.test_pr_auc = []
        self.test_balanced_accuracy = []
        self.permuted_auc = []

        self.p_value = np.nan
        self.class_balance = np.nan
        self.model_name = model.__class__.__name__

    def __repr__(self) -> str:
        cv_name = self._get_cv().__class__.__name__
        parts = [
            f'{self.model_name}',
            f'{self.n_splits}-fold {cv_name}',
            f'n_permutations={self.n_permutations}',
            f'random_state={self.random_state}',
            f'fitted={self.fitted}',
        ]

        if self.fitted:
            parts.extend([
                f'mean_test_roc_auc={self.mean_test_roc_auc:.3f}',
                f'mean_test_pr_auc={self.mean_test_pr_auc:.3f}',
                f'mean_test_balanced_accuracy={self.mean_test_balanced_accuracy:.3f}',
                f'class_balance={self.class_balance:.3f}',
            ])
            if self.n_permutations > 0:
                parts.append(f'p_value={self.p_value:.3f}')

        return f"DecoderCV({', '.join(parts)})"

    def reset(self):
        """Clear stored results from a previous fit.

        Returns
        -------
        DecoderCV
            The same instance, with all metric containers reset to empty values.
        """
        self.train_roc_auc = []
        self.test_roc_auc = []
        self.test_pr_auc = []
        self.test_balanced_accuracy = []
        self.permuted_auc = []
        self.p_value = np.nan
        self.class_balance = np.nan
        return self

    def _get_cv(self):
        """Return the configured cross-validation splitter.

        Returns
        -------
        object
            A splitter with a `.split(X, y)` method. This is either `self.cv`
            or a default `StratifiedKFold` instance built from `cv_splits` and
            `random_state`.
        """
        if self.cv is None:
            return StratifiedKFold(n_splits=self.cv_splits,shuffle=True,random_state=self.random_state,)
        return self.cv

    def _get_scores(self, model, X):
        """Extract continuous decision scores from a fitted estimator.

        Parameters
        ----------
        model : sklearn estimator
            Fitted binary classifier.
        X : np.ndarray
            Array of shape `(n_samples, n_features)` to score.

        Returns
        -------
        np.ndarray
            Continuous scores of shape `(n_samples,)`, taken from
            `predict_proba(..., class 1)` or `decision_function`.
        """
        if hasattr(model, 'predict_proba'):
            score = model.predict_proba(X)[:, 1]
        elif hasattr(model, 'decision_function'):
            score = model.decision_function(X)
        else:
            raise ValueError('Model must implement predict_proba or decision_function.')
        return score

    def _run_fold_loop(self, X, y, store_train=False, store_test=False):
        """Run one full cross-validation pass on a fixed label vector.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape `(n_samples, n_features)`.
        y : np.ndarray
            Binary label vector of shape `(n_samples,)`.
        store_train : bool, default=False
            If `True`, save training-fold ROC AUC values onto the instance.
        store_test : bool, default=False
            If `True`, save held-out-fold ROC AUC, PR AUC, and balanced
            accuracy values onto the instance.

        Returns
        -------
        dict[str, list[float]]
            Dictionary with fold-level metrics:
            `train_roc_auc`, `test_roc_auc`, `test_pr_auc`,
            `test_balanced_accuracy`.
        """
        cv = self._get_cv()
        fold_results = {
            'train_roc_auc': [],
            'test_roc_auc': [],
            'test_pr_auc': [],
            'test_balanced_accuracy': [],
        }

        for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            if self.verbose:
                print(f'Fold {i}')

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])

            model = clone(self.model)
            model.fit(X_train, y[train_idx])

            train_score = self._get_scores(model, X_train)
            test_score = self._get_scores(model, X_test)
            test_pred = model.predict(X_test)

            fold_results['train_roc_auc'].append(roc_auc_score(y[train_idx], train_score))
            fold_results['test_roc_auc'].append(roc_auc_score(y[test_idx], test_score))
            fold_results['test_pr_auc'].append(average_precision_score(y[test_idx], test_score))
            fold_results['test_balanced_accuracy'].append(balanced_accuracy_score(y[test_idx], test_pred))

        if store_train:
            self.train_roc_auc = fold_results['train_roc_auc']
        if store_test:
            self.test_roc_auc = fold_results['test_roc_auc']
            self.test_pr_auc = fold_results['test_pr_auc']
            self.test_balanced_accuracy = fold_results['test_balanced_accuracy']

        return fold_results

    def fit(self, X, y):
        """Evaluate the configured decoder on binary labels under CV.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape `(n_samples, n_features)`.
        y : np.ndarray
            Binary label vector of shape `(n_samples,)`. Positive class is
            assumed to be encoded as `1`. 

        Returns
        -------
        DecoderCV
            The same instance after populating fold-level metrics, summary
            properties, and optional permutation-test results.

        Notes
        -----
        Features are standardized separately within each fold using statistics
        computed on the training partition only.
        """
        self.reset()
        self.class_balance = float(np.mean(y == 1))

        if self.verbose:
            print('== Compute actual AUC ==')
        self._run_fold_loop(X, y, store_train=True, store_test=True)

        if self.n_permutations > 0:
            if self.verbose:
                print('== Compute permuted AUC ==')

            rng = np.random.RandomState(self.random_state)
            for _ in tqdm(
                range(self.n_permutations),
                desc='Permutations',
                leave=False,
                disable=not self.verbose,
            ):
                y_perm = rng.permutation(y)
                fold_results = self._run_fold_loop(X, y_perm, store_train=False, store_test=False)
                self.permuted_auc.append(float(np.mean(fold_results['test_roc_auc'])))

            self.p_value = (np.sum(np.array(self.permuted_auc) >= self.mean_test_roc_auc) + 1) / (len(self.permuted_auc) + 1)

        return self

    @property
    def mean_train_roc_auc(self) -> float:
        return float(np.mean(self.train_roc_auc)) if self.train_roc_auc else np.nan

    @property
    def mean_test_roc_auc(self) -> float:
        return float(np.mean(self.test_roc_auc)) if self.test_roc_auc else np.nan

    @property
    def mean_test_pr_auc(self) -> float:
        return float(np.mean(self.test_pr_auc)) if self.test_pr_auc else np.nan

    @property
    def mean_test_balanced_accuracy(self) -> float:
        return float(np.mean(self.test_balanced_accuracy)) if self.test_balanced_accuracy else np.nan

    @property
    def n_splits(self) -> int | None:
        if self.cv is None:
            return self.cv_splits
        if hasattr(self.cv, 'n_splits'):
            return self.cv.n_splits
        return None

    @property
    def fitted(self) -> bool:
        return len(self.test_roc_auc) > 0

def random_orthonormal_basis(n_features: int, n_components: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a random orthonormal basis with matched dimensionality.

    Parameters
    ----------
    n_features : int
        Number of rows in the basis, usually the number of observed features.
    n_components : int
        Number of orthonormal columns to return.
    rng : np.random.Generator
        NumPy random generator used to sample the basis.

    Returns
    -------
    np.ndarray
        Array of shape `(n_features, n_components)` with orthonormal columns.
    """
    basis = rng.standard_normal((n_features, n_components))
    q, _ = np.linalg.qr(basis)
    return q[:, :n_components]

def get_status_features(s, signal: str, rng: np.random.Generator | None = None) -> np.ndarray:
    """Build one feature matrix for within-task on/off decoding.

    Parameters
    ----------
    s : object of class Session from data_loader.py
        Session-like object used throughout this codebase. It must expose the
        fields and methods used here, including `find_subspace()`, `Y`,
        `Y_norm`, and `FA`.
    signal : {'raw', 'norm', 'latent', 'latent_rand', 'ortho_rand'}
        Name of the feature representation to construct.
    rng : np.random.Generator or None, default=None
        Random generator used for randomized control representations. Required
        for `latent_rand` and `ortho_rand`.

    Returns
    -------
    np.ndarray
        Feature matrix `X` of shape `(n_samples, n_features_or_components)`
        used as input to `DecoderCV.fit`.
    """
    s.find_subspace()

    if signal == 'raw': return s.Y

    if signal == 'norm': return s.Y_norm

    if signal == 'latent':
        # Match the dimensionality of both control representations.
        return s.FA.fa_transform(X=s.Y_norm)[:, :s.FA.d_shared]

    if rng is None:
        raise ValueError('A random generator is required for randomized signal types.')

    if signal == 'latent_rand':
        Y_rand = io.permute_columns_independently(s.Y_norm, rng=rng)
        FA_rand = decomposition.FAResults().fa_fit(
            Y_rand, n_components=s.FA.d_shared, max_iter=int(1e5)
        )
        return FA_rand.fa_transform(X=s.Y_norm)

    if signal == 'ortho_rand':
        basis = random_orthonormal_basis(s.Y_norm.shape[1], s.FA.d_shared, rng)
        return s.Y_norm @ basis

    raise ValueError(f'Unknown signal: {signal}')

def decode_status_single_session(
    s: io.Session,
    signal: str,
    decoder_kwargs: dict,
    rng: np.random.Generator | None = None,
    n_repeats: int = 1,
    show_progress: bool = False,
):
    """Decode on/off state for one session and return summary metrics.

    Parameters
    ----------
    s : object of class Session from data_loader.py
    signal : {'raw', 'norm', 'latent', 'latent_rand', 'ortho_rand'}
        Feature representation to decode from.
    decoder_kwargs : dict
        Keyword arguments passed into `DecoderCV(...)`.
    rng : np.random.Generator or None, default=None
        Random generator used for randomized control representations.
    n_repeats : int, default=1
        Number of randomized feature draws for `latent_rand` and `ortho_rand`.
    show_progress : bool, default=False
        If `True`, show a progress bar across randomized repeats for
        `latent_rand` and `ortho_rand`.

    Returns
    -------
    dict[str, float | int] or None
        Dictionary of summary metrics for one session. Returns `None` if the
        session has no valid on/off labels.

        Returned keys for standard signals:
        - `actual_auc` : float
        - `pr_auc` : float
        - `balanced_accuracy` : float
        - `class_balance` : float

        Additional keys for randomized signals:
        - `null_auc_std` : float
        - `n_repeats` : int
        - `roc_auc_repeats` : list[float]
        - `pr_auc_repeats` : list[float]
        - `balanced_accuracy_repeats` : list[float]
    """
    y = get_on_off(s, max_gap=1.0)
    if y is None:
        return None

    if signal in ['raw', 'norm', 'latent']:
        X = get_status_features(s, signal, rng=rng)
        decoder = DecoderCV(**decoder_kwargs).fit(X, y)
        return {
            'actual_auc': decoder.mean_test_roc_auc,
            'pr_auc': decoder.mean_test_pr_auc,
            'balanced_accuracy': decoder.mean_test_balanced_accuracy,
            'class_balance': decoder.class_balance,
        }

    if signal in ['latent_rand', 'ortho_rand']:
        roc_aucs, pr_aucs, bal_accs = [], [], []
        for _ in tqdm(
            range(n_repeats),
            desc=f'{signal} repeats',
            leave=False,
            disable=(n_repeats <= 1) or (not show_progress),
        ):
            X = get_status_features(s, signal, rng=rng)
            decoder = DecoderCV(**decoder_kwargs).fit(X, y)
            roc_aucs.append(decoder.mean_test_roc_auc)
            pr_aucs.append(decoder.mean_test_pr_auc)
            bal_accs.append(decoder.mean_test_balanced_accuracy)

        return {
            'actual_auc': float(np.mean(roc_aucs)),
            'null_auc_std': float(np.std(roc_aucs, ddof=1)) if len(roc_aucs) > 1 else 0.0,
            'pr_auc': float(np.mean(pr_aucs)),
            'balanced_accuracy': float(np.mean(bal_accs)),
            'class_balance': float(np.mean(y == 1)),
            'n_repeats': n_repeats,
            'roc_auc_repeats': roc_aucs,
            'pr_auc_repeats': pr_aucs,
            'balanced_accuracy_repeats': bal_accs,
        }

    raise ValueError(f'Unknown signal: {signal}')

def decode_status_within_task(
        df: pd.DataFrame,
        signal: str = 'raw', 
        n_permutations: int = 500, 
        n_null_repeats: int = 1,
        model = LogisticRegression(solver='liblinear'),
        show_progress: bool = True,
    ):
    """Decode on/off state separately within each session in a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Session-level dataframe. It must contain a `session_class` column and a
        `Task` column and a `session_key` column.
    signal : {'raw', 'norm', 'latent', 'latent_rand', 'ortho_rand'}, default='raw'
        Feature representation to decode within each session.
    n_null_repeats : int, default=1
        Number of randomized feature draws for `latent_rand` and `ortho_rand`.
    n_permutations : int, default=500
        Number of label permutations passed to `DecoderCV` for each session.
    model : sklearn estimator, default=LogisticRegression(solver='liblinear')
        Binary classifier used inside each decoder.
    show_progress : bool, default=True
        If `True`, show a progress bar across randomized repeats for
        `latent_rand` and `ortho_rand`. No progress bar is shown for the outer
        session loop.

    Returns
    -------
    pd.DataFrame
        Dataframe containing copied session-identifying columns and decoding
        summaries. Expected result columns include:
        - `task` : str
        - `actual_auc` : float
        - `pr_auc` : float
        - `balanced_accuracy` : float
        - `class_balance` : float
        Randomized signal types may also include:
        - `null_auc_std` : float
        - `n_repeats` : int
        - `roc_auc_repeats` : list[float]
        - `pr_auc_repeats` : list[float]
        - `balanced_accuracy_repeats` : list[float]
    """

    df_res = df[['session_key', 'session_class']].copy()
    df_res['task'] = df.Task.str.lower()
    results = []
    rng = np.random.default_rng(0)
    decoder_kwargs = dict(
        model=model,
        cv_splits=10,
        n_permutations=n_permutations,
        random_state=0,
        verbose=False,
    )

    for i in range(len(df)):
        s = df.iloc[i].session_class
        print(s)
        result = decode_status_single_session(
            s,
            signal=signal,
            decoder_kwargs=decoder_kwargs,
            rng=rng,
            n_repeats=n_null_repeats,
            show_progress=show_progress,
        )
        results.append(result if result is not None else {})
    
    return pd.concat([df_res, pd.DataFrame(results, index=df.index)], axis=1)

def get_common_channel_filenames(df: pd.DataFrame,
                                 brain_areas: list[str] = ['HPC', 'ACC'],
                                 min_fr: float = MIN_FR,
                                 must_have_cluster: bool = True) -> list[str]:
    common, ordered = None, None
    for i in range(len(df)):
        s = df.iloc[i].session_class
        s.choose_channels(brain_areas=brain_areas,min_fr=min_fr,must_have_cluster=must_have_cluster)
        filenames = s.chosen_chan.filename.tolist()
        chan_set = set(filenames)
        if common is None:
            common = chan_set
            ordered = filenames
        else:
            common = common & chan_set
    if common is None: return []
    return [filename for filename in ordered if filename in common]

def get_common_channel_indices(s, common_filenames: list[str]) -> np.ndarray:
    df_chan = s.chosen_chan.reset_index()[['index', 'filename']]
    return df_chan.set_index('filename').loc[common_filenames, 'index'].to_numpy()

def prepare_decoding_lme(decoding, metadata):
    """Join patient identity from metadata; audit and drop only missing AUCs.

    Pass df_gap, or io.load_sessions_csv() to avoid loading neural recordings.
    Randomized draws must already be averaged to one row/session/representation.
    Returns the model dataframe, per-session coverage, and excluded AUC rows.
    """
    meta = metadata[['session_key', 'Patient', 'Task']].drop_duplicates().copy()
    if meta.isna().any().any() or meta.session_key.duplicated().any():
        raise ValueError('Each metadata session_key must identify exactly one patient and task.')
    frame = decoding[['session_key', 'task', 'group', 'actual_auc']].merge(
        meta, on='session_key', how='left', validate='many_to_one', indicator=True,
    )
    if not frame['_merge'].eq('both').all():
        missing = frame.loc[frame['_merge'] != 'both', 'session_key'].unique()
        raise ValueError(f'Sessions missing from metadata: {missing}')
    frame['task'] = frame.task.str.lower()
    if not frame.task.eq(frame.Task.str.lower()).all():
        raise ValueError('Decoding task labels disagree with session metadata.')
    labels = {'latent': 'task_conditioned', 'latent_rand': 'shuffle', 'ortho_rand': 'random'}
    frame['representation'] = frame.group.map(labels)
    if frame.representation.isna().any() or not frame.task.isin(TASK_NAMES).all():
        raise ValueError('Unknown or missing representation/task label.')
    frame = frame.rename(columns={'Patient': 'patient', 'session_key': 'session'})
    frame = frame[['patient', 'session', 'task', 'representation', 'actual_auc']].copy()
    if frame.duplicated(['session', 'representation']).any():
        raise ValueError('Duplicate session × representation rows; do not pool randomized draws.')
    frame['actual_auc'] = pd.to_numeric(frame.actual_auc, errors='raise')
    observed = frame.actual_auc.dropna()
    if not observed.between(0, 1).all():
        raise ValueError('Nonmissing AUCs must be finite and between zero and one.')
    coverage = frame.groupby(['patient', 'session', 'task'], sort=False).agg(
        n_representation_rows=('representation', 'nunique'),
        n_representations=('actual_auc', 'count'),
    ).reset_index()
    excluded = frame.loc[frame.actual_auc.isna()].copy()
    frame = frame.dropna(subset=['actual_auc']).reset_index(drop=True)
    frame['representation'] = pd.Categorical(frame.representation, categories=list(labels.values()))
    frame['task'] = pd.Categorical(frame.task, categories=TASK_NAMES)
    if frame.empty:
        raise ValueError('No nonmissing AUC observations remain.')
    return frame, coverage, excluded

def fit_decoding_lme(df_lme, method='lbfgs'):
    """REML: representation + task, with patient and nested-session intercepts.

    Fixed-effect inference uses asymptotic Wald z tests, not finite-df t tests.
    Capture warnings despite the plotting script's global warning filter. Never
    substitute an independent-observations model when fitting fails.
    """
    frame = df_lme.copy()
    frame['representation'] = pd.Categorical(
        frame.representation, categories=['task_conditioned', 'shuffle', 'random'],
    )
    frame['task'] = pd.Categorical(frame.task, categories=TASK_NAMES)
    if frame.isna().any().any() or frame.duplicated(['session', 'representation']).any():
        raise ValueError('Use prepare_decoding_lme to obtain complete, unique model rows.')
    if (frame.groupby('session')[['patient', 'task']].nunique() != 1).any().any():
        raise ValueError('Each session must belong to exactly one patient and task.')
    model = smf.mixedlm(
        'actual_auc ~ representation + task', data=frame, groups='patient',
        re_formula='1', vc_formula={'session': '0 + C(session)'}, missing='raise',
    )
    if np.linalg.matrix_rank(model.exog) < model.exog.shape[1]:
        raise ValueError('Fixed effects are not identifiable; check representation/task coverage.')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = model.fit(reml=True, method=method, maxiter=2000, full_output=True, disp=False)
    messages = list(dict.fromkeys(str(w.message) for w in caught))
    covariance = result.cov_params().to_numpy()
    variances = pd.Series({
        'patient': result.cov_re.iloc[0, 0],
        'session': result.vcomp[0],
        'residual': result.scale,
    }, name='variance')
    positive_covariance = bool(np.isfinite(covariance).all()
                               and np.linalg.eigvalsh(covariance).min() > 0)
    # The library's absolute 0.01 boundary threshold can flag small AUC variances.
    near_zero = bool((variances / variances.sum() < 1e-6).any())
    boundary_message = 'The MLE may be on the boundary of the parameter space.'
    diagnostics = {
        'method': method, 'reml': True, 'converged': bool(result.converged),
        'n_patients': frame.patient.nunique(), 'n_sessions': frame.session.nunique(),
        'n_observations': len(frame), 'log_likelihood': result.llf,
        'positive_parameter_covariance': positive_covariance,
        'near_zero_variance': near_zero, 'warnings': messages,
        'numerically_valid': bool(result.converged and positive_covariance and not near_zero
                                  and not any(m != boundary_message for m in messages)),
    }
    fixed = pd.DataFrame({
        'coefficient': result.fe_params, 'std_error': result.bse_fe,
        'z': result.tvalues.loc[result.fe_params.index],
        'p_value': result.pvalues.loc[result.fe_params.index],
        'ci_lower': result.conf_int().loc[result.fe_params.index, 0],
        'ci_upper': result.conf_int().loc[result.fe_params.index, 1],
    }).rename_axis('term')
    for message in messages:
        print(f'MixedLM warning: {message}')
    if not diagnostics['numerically_valid']:
        print('Model requires diagnosis before contrasts; no alternative model was fitted.')
    return result, fixed, variances, diagnostics

def decoding_lme_contrasts(result, diagnostics, alpha=0.05):
    """Model-based differences (first minus second), with Holm across 3 tests.

    Confidence intervals are pointwise, unadjusted Wald intervals. Also return
    a joint 2-df Wald chi-square test of the representation fixed effect.
    """
    if not diagnostics['numerically_valid']:
        raise RuntimeError('Resolve model convergence/singularity before running contrasts.')
    names = result.fe_params.index
    contrast = np.zeros((3, len(names)))
    contrast[0, names.get_loc('representation[T.shuffle]')] = -1
    contrast[1, names.get_loc('representation[T.random]')] = -1
    contrast[2] = contrast[1] - contrast[0]
    pairs = result.t_test(contrast, use_t=False).summary_frame(alpha=alpha)
    pairs.index = ['task_conditioned - shuffle', 'task_conditioned - random', 'shuffle - random']
    pairs = pairs.rename_axis('contrast').rename(columns={
        'coef': 'difference', 'std err': 'std_error', 'P>|z|': 'p_value',
        'Conf. Int. Low': 'ci_lower', 'Conf. Int. Upp.': 'ci_upper',
    })
    pairs['significant_holm'], pairs['p_holm'], _, _ = multipletests(
        pairs.p_value, alpha=alpha, method='holm',
    )
    joint = np.zeros((2, len(result.params)))
    
    joint[:, :len(names)] = contrast[:2]
    test = result.wald_test(joint, use_f=False, scalar=True)
    omnibus = pd.Series({'chi2': float(test.statistic), 'df': 2, 'p_value': float(test.pvalue)})
    return pairs, omnibus

#%% Read data 

df_gap = read_all_sessions(max_gap_dur_secs=0.1, bin_size=100, get_subspace=True, event_mode='all')
df_gap = df_gap[(df_gap.session_class!='ignore') & ~(df_gap.filename.isin(EXCLUDED_SESSIONS))].reset_index(drop=True)
df_all = read_all_sessions(max_gap_dur_secs=1, bin_size=100, get_subspace=True, event_mode='events')

#%% Decode status within task with normed spikes using logistic regression (Fig 2G, tasks)

df_res = decode_status_within_task(
    df=df_gap, 
    signal='norm', 
    n_permutations=0
)
fig, ax = plt.subplots(figsize=(2.5,2.5), dpi=300)
sns.boxplot(data=df_res, x='task', y='actual_auc', ax=ax, linewidth=LW, whis=(0, 100), fill=False, color='k', fliersize=0)
sns.stripplot(data=df_res,x='task',y='actual_auc',hue='task',hue_order=TASK_NAMES,palette=TASK_CLR,ax=ax,s=5,edgecolor='k',linewidth=LW,dodge=False,jitter=0.05, legend=False)
plt.ylabel('AUC')
plt.title(f'Decoding task status w normed spikes', fontsize=5)
plt.xticks([0,1,2], [w.capitalize() for w in TASK_NAMES])
plt.ylim([0.5,None])
plt.xlabel('')
ax.legend(title=None, frameon=False, fontsize=6, loc='best')
sns.despine(trim=False)
plt.tight_layout()
save_dir = os.path.join('plots','decoding')
finish_plot(filename=f'decode_status_normed_spikes', save_dir=save_dir, savefig=SAVEFIG)
#%% Decode identity across tasks with raw spikes using logistic regression (Fig 2I, tasks)

df_pairs = get_paired_df(df_all)
for i in tqdm(range(len(df_pairs)), desc='decode_identity'):
    row = df_pairs.iloc[i]
    task1, task2, common = row.task_1, row.task_2, row.common
    X1 = task1.find_y(np.where(task1.df_chan.filename.isin(common))[0]).get_spikes_mu_sd().Y
    X2 = task2.find_y(np.where(task2.df_chan.filename.isin(common))[0]).get_spikes_mu_sd().Y
    y1 = np.zeros(X1.shape[0])
    y2 = np.ones(X2.shape[0])
    X = np.concatenate([X1, X2], axis=0)
    y = np.concatenate([y1, y2], axis=0)

    decoder = DecoderCV(
        model=LogisticRegression(solver='liblinear'),
        cv_splits=5,
        random_state=0,
        verbose=False,
    ).fit(X, y)

    df_pairs.at[i, 'decoder'] = decoder
    df_pairs.at[i, 'actual_auc'] = decoder.mean_test_roc_auc
    df_pairs.at[i, 'pr_auc'] = decoder.mean_test_pr_auc
    df_pairs.at[i, 'balanced_accuracy'] = decoder.mean_test_balanced_accuracy
    df_pairs.at[i, 'class_balance'] = decoder.class_balance

fig, ax = plt.subplots(figsize=(2.5,3.2), dpi=300)
sns.boxplot(data=df_pairs.dropna(), x='task_pair', y='actual_auc', hue='task_pair', fill=False, ax=ax, linewidth=LW, legend=None, color='k', whis=(0, 100))
sns.stripplot(data=df_pairs.dropna(), x='task_pair', y='actual_auc', hue='task_pair', palette=TASK_PAIRS_CLR,ax=ax, s=5, edgecolor='k', linewidth=LW, dodge=False, legend=None)
ax.set_ylabel('AUC')
ax.set_xlabel('')
plt.xticks([0,1,2], [f'{TASK_KEY_R[pair[0]].capitalize()} / {TASK_KEY_R[pair[1]].capitalize()}' for pair in TASK_PAIRS], rotation=45)
plt.ylim([0.5,None])
plt.tight_layout()
sns.despine(trim=False)
save_dir = os.path.join('plots','decoding')
finish_plot(filename=f'decode_identity_across_task_raw_spikes_linear', save_dir=save_dir, savefig=SAVEFIG)
plt.show()

#%% Decode status within task with latent factors using logistic regression (computing only for Fig 3H)
RERUN = False
if RERUN:
    n_null_repeats = 30
    df_ = []
    for signal in ['latent', 'latent_rand', 'ortho_rand']:
        df_res = decode_status_within_task(
            df=df_gap,
            signal=signal,
            n_permutations=0,
            n_null_repeats=n_null_repeats,
        )
        df_res['group'] = signal
        df_.append(df_res)
    df_ = pd.concat(df_).reset_index(drop=True)
    model_save_dir = os.path.join('models', 'decoding')
    os.makedirs(model_save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d-%H%M')
    df_.to_pickle(os.path.join(model_save_dir, f'decode_status_latent_{n_null_repeats}_{timestamp}.pkl'))
#%% Decode status with different subspace activity (Fig 3H, subspace)

# read pickle df
model_save_dir = os.path.join('models', 'decoding')
df_ = pd.read_pickle(os.path.join(model_save_dir, f'decode_status_latent_30_20260408-0449.pkl'))

fig, ax = plt.subplots(figsize=(2.5,3), dpi=300)
sns.boxplot(data=df_, x='group', y='actual_auc', ax=ax, linewidth=LW,
            whis=(0, 100), fill=False, color='k', fliersize=0)
sns.swarmplot(data=df_,x='group',y='actual_auc',hue='task',hue_order=TASK_NAMES,
              palette=TASK_CLR,ax=ax,s=4,edgecolor='k',linewidth=LW,dodge=False, legend=False)
plt.ylabel('AUC')
plt.title(f'Decoding task status w subspace activity', fontsize=5)
plt.xlabel('')
plt.xticks([0,1,2], [
    'Task-conditioned\nsubspace',
    'Null \nsubspace',
    'Random\nsubspace'
    ], rotation=45, ha='right', rotation_mode='anchor')
ax.legend(title=None, frameon=False, fontsize=6, loc='best')
sns.despine(trim=False)
plt.tight_layout()
save_dir = os.path.join('plots','decoding')
finish_plot(filename=f'decode_status_latent', save_dir=save_dir, savefig=SAVEFIG)

## Stats using LME
# step 1: validate saved AUCs and join patient metadata (no decoding)
# df_gap can also be passed here; the CSV loader only reads session metadata.
df_lme, lme_coverage, lme_excluded = prepare_decoding_lme(df_, io.load_sessions_csv())
print(lme_coverage.to_string(index=False))
print(f'Excluded {len(lme_excluded)} missing AUC rows.')
print(f'{df_lme.patient.nunique()} patients, {df_lme.session.nunique()} sessions, {len(df_lme)} observations')

# step 2: fit and inspect diagnostics before interpreting contrasts
result, lme_fixed, lme_variances, lme_diagnostics = fit_decoding_lme(df_lme)
print(lme_fixed.to_string())
print(lme_variances.to_string())
print(lme_diagnostics)

# step 3: model-based contrasts, Holm correction across the three pairs
lme_pairs, lme_omnibus = decoding_lme_contrasts(result, lme_diagnostics)
print(lme_omnibus.to_string())
print(lme_pairs.to_string())


# %%
