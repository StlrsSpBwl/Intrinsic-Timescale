import numpy as np
from scipy.special import digamma,polygamma
from scipy.optimize import brentq
from scipy.stats import gamma as gamma_dist
import warnings

### Core HMM-Gamma Class
class HMMGamma:
    """
    Hidden Markov Model with Gamma-distributed emissions

    Input:
    n_states: number of hidden states (example: 3 for alert/drowsy/sleepy)
    n_features: number of observed variables (example: how many timescales/physiological elements we are putting in)
    max_iter:int (maximum EM iterations for Baum-Welch)
    tol:float (convergence threshold for EM)
    n_restarts:int (number of random initializations to avoid local minima)
    random_state:int (random seed for reproducibility)
    """
    def __init__(self,n_states=3,n_features=1,max_iter=200,tol=1e-6,
                 n_restarts=10,random_state=None):
        self.n_states = n_states
        self.n_features = n_features
        self.max_iter = max_iter
        self.tol = tol
        self.n_restarts = n_restarts
        self.rng = np.random.RandomState(random_state)
        # Initialize model parameters
        self.pi = None  # Initial state probabilities
        self.A = None
        self.shape = None
        self.scale = None
        # Emission probability
        def _log_emission(self,X):
            """"
            Compute log(P_t|state k) for all t,k
            Parameters:
            X: ndarray(T,D); Observation sequence, all values must be >0

            Returns:
            log_B: ndarray(T,K); log emission probabilities for each time t and state k
            """
            T,D= X.shape
            K=self.n_states
            log_B= np.zeros((T,K))
            for k in range(K):
                for d in range(D):
                    # Gamma log-pdf: (a-1)*log(x)-x/scale - a*log(scale) - gammaln(a)
                    log_B[:,k]+=gamma_dist.logpdf(
                        X[:,d],a=self.shape[k,d],scale=self.scale[k,d]
                    )
            return log_B
    
    # forward-backward algorithm (using log-spae for numerical stability)
    def _forward(self,log_B):
        """
        Forward pass in log space

        Returns
        log_alpha: ndarray(T,K);
        log_likelihood: float
        """
        T,K = log_B.shape
        log_alpha = np.full((T,K),-np.inf)
        # when t is 0
        log_alpha[0]=np.log(self.pi+1e-300)+log_B[0]

        # t = 1... T-1
        log_A =np.log(self.A+1e-300)
        for t in range(1,T):
            for k in range(K):
                log_alpha[t,k]=(
                    _logsumexp(log_alpha[t-1]+log_A[:,k])+log_B[t,k]
                )
        log_likelihood = _logsumexp(log_alpha[-1])
        return log_alpha,log_likelihood
    def _backward(self,log_B):
        """
        backward pass in log space
        
        Returns: log_beta: ndarray(T,K)
        """
        T,K = log_B.shape
        log_beta = np.full((T,K),-np.inf)
        log_beta[-1]=0.0
        log_A = np.log(self.A+1e-300)
        for t in range(T-2,-1,-1):
            for k in range(K):
                log_beta[t,k]=_logsumexp(
                    log_A[k,:]+log_B[t+1,:]+log_beta[t+1,:]
                )
        return log_beta
    def _compute_posteriors(self,log_alpha,log_beta):
        """
        Compute state posteriors gamma_t(k) = P(state_t=k|observations) in log space

        Returns: 
        gamma:ndarray(T,K); state posteriors for each time t and state k
        """
        log_gamma = log_alpha + log_beta
        # Normalize per time step
        log_gamma -= _logsumexp_axis(log_gamma,axis=1,keepdims=True)
        return np.exp(log_gamma)
    
    def _compute_xi(self,log_alpha,log_beta,log_B):
        """"
        Compute transition posteriors xi_t(i,j) = P(s_t=i,s_{t+1}=j|obs)

        Returns
        xi: ndarray(T-1,K,K)
        """
        T, K = log_B.shape
        log_A = np.log(self.A+1e-300)
        log_xi = np.full((T-1,K,K),-np.inf)
        for t in range(T-1):
            for i in range(K):
                for j in range(K):
                    log_xi[t,i,j]=(
                        log_alpha[t,i]+log_A[i,j]+log_B[t+1,j]+log_beta[t+1,j]
                    )
            # Normalize
            log_xi[t]-=_logsumexp(log_xi[t].ravel())
        return np.exp(log_xi)

# EM algorithm-M step: update Gamma emission parameters
    def _m_step_emissions(self,X,gamma):
        """
        Update shape and scale parameters for each state's gamma emissions
        Use maximum likelihood via the digamma equations from (Zhang et al)
        log(k)-digamma(k)=log(x_bar)-log_x_bar
        where x_bar is the weighted average of observations for that state
        """
        T,D = X.shape
        K = self.n_states
        log_X = np.log(np.maximum(X,1e-300))  # Avoid log(0)
        for k in range(K):
            w=gamma[:,k]
            w_sum = w.sum()+1e-300
            for d in range(D):
                # Weighted statistics
                x_bar = np.dot(w,X[:,d])/w_sum
                log_x_bar = np.dot(w,log_X[:,d])/w_sum
                # s = log(x_bar)-log_x_bar
                s = np.log(x_bar)-log_x_bar
                s = max(s,1e-10)
                # solve:log(k)-digamma(k)=s for k >0
                try:
                    def objective(k):
                        return np.log(k)-digamma(k)-s
                    # initial bracket: for small s, k is large; for large s, k is small
                    shape_k = brentq(objective,1e-6,1e6,xtol=1e-10)
                except (ValueError,RuntimeError):
                    var_x = np.dot(w,X[:,d]**2)/w_sum
                    var_x = max(var_x,1e-10)
                    shape_k = x_bar**2/var_x
                shape_k=max(shape_k,1e-4)
                scale_k = x_bar/shape_k
                self.shape[k,d]=shape_k
                self.scale[k,d]=scale_k

# Initialization and fitting
    def _initialize(self,X):
        T,D = X.shape
        K = self.n_states
        # Initial state distribution: uniform
        self.pi = np.ones(K)/K
        # Transition matrix
        self.A = np.full((K,K),0.05/(K-1))
        np.fill_diagonal(self.A,0.95)
        # Add small noise
        self.A+=self.rng.uniform(0,0.002,size=(K,K))
        self.A /= self.A.sum(axis=1,keepdims=True)
        # Emission parameters:
        row_means = X.mean(axis=1)
        quantiles = np.linspace(0,100,K+1)
        self.shapes = np.zeros((K,D))
        self.scales = np.zeros((K,D))
        for k in range(K):
            lo = np.percentile(row_means,quantiles[k])
            hi = np.percentile(row_means,quantiles[k+1])
            mask = (row_means>=lo) & (row_means<hi)
            if mask.sum()<5:
                mask = self.rng.choice(T,size=max(5,T//K),replace=True)
            for d in range(D):
                subset = X[mask,d] if isinstance(mask,np.ndarray) and mask.dtype == bool else X[mask,d]
                subset = subset[subset>0]
                if len(subset)<2:
                    subset = X[:,d][X[:,d]>0]
                m = subset.mean()
                v= subset.var()
                v = max(v,1e-10)
                self.shapes[k,d]=m**2/v
                self.scales[k,d]=v/m
    def fit(self,X,verbose=False):
        """
        Fit the HMM-Gamma model to data X using the Baum-Welch

        Parameters:
        X: ndarray(T,D); observed data, all values must be >0
        verbose:bool

        Returns: self
        """
        X =self._validate_input(X)
        T,D = X.shape
        assert D == self.n_features, f"Expected {self.n_features} features, got {D}"
        best_ll = - np.inf
        best_params = None
        for restart in range(self.n_restarts):
            self._initialize(X)
            ll_prev = -np.inf
            for iteration in range(self.max_iter):
                # E-step
                log_B = self._log_emission(X)
                log_alpha,ll = self._forward(log_B)
                log_beta = self._backward(log_B)
                if np.isnan(ll) or np.isinf(ll):
                    break
                gamma = self._compute_posteriors(log_alpha,log_beta)
                xi = self._compute_xi(log_alpha,log_beta,log_B)
                # M-step: transition
                self.pi = gamma[0]/gamma[0].sum()
                xi_sum = xi.sum(axis=0)
                self.A=xi_sum/(xi_sum.sum(axis=1,keepdims=True)+1e-300)
                # M-step: emissions
                self._m_step_emissions(X,gamma)
                # Check convergence
                if verbose:
                    print(f"Restart {restart}, Iter {iteration}, Log-Likelihood: {ll:.2f}")
                if abs(ll-ll_prev)<self.tol:
                    break
                ll_prev = ll
            if ll>best_ll and not np.isnan(ll):
                best_ll = ll
                best_params = {
                    'pi':self.pi.copy(),
                    'A':self.A.copy(),
                    'shapes':self.shapes.copy(),
                    'scales':self.scales.copy(),
                    'll':ll,
                    'n_iter':iteration,
                    'restart':restart
                }
        if best_params is not None:
            self.pi = best_params['pi']
            self.A = best_params['A']
            self.shapes = best_params['shapes']
            self.scales = best_params['scales']
            self.train_ll = best_params['ll']
            if verbose:
                print(f"\nBest LL = {best_params['ll']:.2f}"
                      f"(restart {best_params['restart']},"
                      f"iter{best_params['n_iter']})")
        else:
            warnings.warn("All restarts failed. Model parameters maybe invalid.")
        
        self._order_states()
        return self
    def _order_states(self):
        means = (self.shapes*self.scales).mean(axis=1)
        order = np.argsort(means)
        self.pi = self.pi[order]
        self.A = self.A[order][:,order]
        self.shapes = self.shapes[order]
        self.scales = self.scales[order]
# Inference
    def decode(self,X):
        """"
        Viterbi decoding: find the single most likely state sequence
        Input: X as ndarray(T,D) or (T,)

        Returns: 
        states: ndarray(T,) of int (most likely state at each time step)
        log_prob: float (log probability of the best path)
        """
        X = self._validate_input(X)
        log_B = self._log_emission(X)
        T,K = log_B.shape
        log_A = np.log(self.A+1e-300)
        log_delta = np.zeros((T,K))
        psi = np.zeros((T,K),dtype=int)
        # Initialize
        log_delta[0]=np.log(self.pi+1e-300)+log_B[0]
        # Recursion
        for t in range(1,T):
            for k in range(K):
                candidates = log_delta[t-1]+log_A[:,k]
                psi[t,k]=np.argmax(candidates)
                log_delta[t,k]=candidates[psi[t,k]]+log_B[t,k]
        # Backtrack
        states = np.zeros(T,dtype=int)
        states[-1]=np.argmax(log_delta[-1])
        log_prob = log_delta[-1,states[-1]]
        for t in range(T-2,-1,-1):
            states[t] = psi[t+1,states[t+1]]
        return states, log_prob
    def predct_proba(self,X):
        """""
        compute state posterior probabilities for each time step
        Input: X as ndarray(T,D) or (T,)
        Returns:
        gamma: ndarray(T,K); P(state_t=k|full observation sequence) for each t, k.
        log_likelihood:float
        """
        X = self._validate_input(X)
        log_B = self._log_emission(X)
        log_alpha,ll=self._forward(log_B)
        log_beta = self._backward(log_B)
        gamma = self._compute_posteriors(log_alpha,log_beta)
        return gamma,ll
    def score(self,X):
        """
        Compute log-likelihood of the data under the model
        Input: X as ndarray(T,D) or (T,)
        Returns: log_likelihood: float
        """
        X = self._validate_input(X)
        log_B = self._log_emission(X)
        _,ll=self._forward(log_B)
        return ll
    
    # State Feature Extraction (for PVT performance mapping)
    def extract_state_features(self,X,hours,target_hours=(9,12,15)):
        """
        Extract state features at specific hours of day for behavioral prediction

        Inputs:
        X: ndarray(T,) or (T, D); Full observation sequence
        hours: ndarray(T,) of int
        target_hours: tuple of int; hours at which PVT sessiosn occur

        Returns:
        features: dict
        Keys are target hours, values are dicts with:
        - 'p_state': (n_days,K) state probabilities at that hour per day
        - 'entropy': (n_days,) transition entropy
        - 'dwell_alert': (n_days,) mean consecutive time in alert state
        """
        X = self._validate_input(X)
        gamma,_=self.predct_proba(X)
        states,_=self.decode(X)
        features = {}
        for h in target_hours:
            mask = hours == h
            indices = np.where(mask)[0]
            if len(indices)==0:
                continue
            # State probabilities at this hour
            p_state = gamma[indices] #(n_occurrences, K)
            # Transition entropy in a window before target hour
            entropies = []
            for idx in indices:
                # look back 120 minutes (2 hours) before this time point
                window_start = max(0,idx-240) # adjust based on sampling rate (current sampling rate is 30 seconds)
                window_states = states[window_start:idx]
                if len(window_states)>1:
                    transitions = np.diff(window_states) != 0
                    p_trans = transitions.mean()
                    # Binary entropy
                    if 0<p_trans<1:
                        ent = -(p_trans*np.log2(p_trans)
                                +(1-p_trans)*np.log2(1-p_trans))
                    else:
                        ent = 0.0
                    entropies.append(ent)
                else:
                    entropies.append(0.0)
            # Mean dwell time in alert statee before target hour
            dwell_times = []
            for idx in indices:
                window_start = max(0,idx-240)
                window_states = states[window_start:idx]
                # Count consecutive alert (state 0) runs
                runs = _count_runs(window_states,target_state=0)
                dwell_times.append(np.mean(runs) if len(runs)>0 else 0.0)
            
            features[h]={
                'p_state':p_state,
                'entropy':np.array(entropies),
                'dwell_alert':np.array(dwell_times),
                'indices':indices
            }
        return features
    # Model selection
    def bic(self,X):
        """Use BIC to select model order"""
        X = self._validate_input(X)
        T = X.shape[0]
        ll = self.score(X)
        K = self.n_states
        D = self.n_features
        # Number of free parameters:
        # pi: K-1, A: K*(K-1), shapes: K*D, scales: K*D*2 (shape+scale per feature)
        n_params = (K-1) + K*(K-1) + 2*K*D
        return -2*ll + n_params*np.log(T)
    def aic(self,X):
        """Use AIC for model selection"""
        X = self._validate_input(X)
        ll=self.score(X)
        K = self.n_states
        D = self.n_features
        n_params = (K-1) + K*(K-1) + 2*K*D
        return -2*ll + 2*n_params
    # Utilities
    def _validate_input(self,X):
        X = np.asarray(X,dtype=float)
        if X.ndim == 1:
            X = X[:,np.newaxis]
        X = np.maximum(X,1e-6)
        return X
    def get_state_means(self): # return mean emission per state per feature
        return self.shapes*self.scales
    
    def get_state_variances(self): # return variance of emission per state per feature
        return self.shapes*(self.scales**2)
    
    def summary(self): # Print model summary
        means = self.get_state_means()
        variance = self.get_state_variances()
        print(f"\nHMM-Gamma Summary ({self.n_states} states, {self.n_features} features)")
        print("="*60)
        print(f"\nInitial State Distribution:")
        for k in range(self.n_states):
            print(f"State {k}: pi={self.pi[k]:.4f}")
        print(f"\nTransition Matrix:")
        for k in range(self.n_states):
            row = " ".join(f"{v:.4f}" for v in self.A[k])
            print(f"State {k}: {row}")
        print(f"\nEmission parameters (mean ± std):")
        for k in range(self.n_states):
            params = []
            for d in range(self.n_features):
                m = means[k,d]
                s = np.sqrt(variance[k,d])
                params.append(f"feat{d}:{m:.2f}±{s:.2f}")
            print(f"State {k}: {', '.join(params)}")
            print(f"  shape={self.shapes[k]}, scale={self.scales[k]}")

# Rolling windows backtesting
def rolling_backtest(X,hours,n_states=3,window_days=7,step_days=1,
                     samples_per_day=2880,verbose=True,**hmm_kwargs):
    """
    Rolling window backtest for HMM-Gamma
    Inputs:
    X: ndarray(T,D) or (T,); full observation sequence
    hours: ndarray(T,) of int; hour of day for each time point
    n_states: int; number of hidden states
    window_days: int; number of days in each training window
    step_days: int; number of days to step forward for each window
    sample_per_day: int; number of samples per day (adjust based on sampling rate or actual number in a day)
    verbose: bool; whether to print progress
    hmm_kwargs: additional kwargs for HMMGamma (e.g. max_iter, tol)

    Returns
    results: dict with keys:
        Each entry contains:
        - 'train_days':(start_day,end_day)
        -'test_days':int
        - 'train_ll':float
        - 'test_ll':float
        - 'test_gamma':ndarray(samples_per_day,K) posterior probs
        - 'test_states':ndarray(samples_per_day,) Viterbi states
        - 'model': fitted HMMGamma
    """
    X = np.asarray(X,dtype=float)
    if X.ndim == 1:
        X = X[:,np.newaxis]
    T,D= X.shape
    n_days = T//samples_per_day
    window_samples = window_days*samples_per_day
    step_samples = step_days*samples_per_day
    test_samples = samples_per_day
    results = []
    for start in range(0,T-window_samples-test_samples+1,step_samples):
        train_end = start+window_samples
        test_end = train_end+test_samples
        if test_end>T:
            break
        X_train = X[start:train_end]
        X_test = X[train_end:test_end]
        train_day_start = start//samples_per_day
        train_day_end = train_end//samples_per_day
        test_day = train_end//samples_per_day
        if verbose:
            print(f"Training days {train_day_start}-{train_day_end-1}, Testing day {test_day}")
        
        # Fit model
        model = HMMGamma(n_states=n_states,n_features=D,**hmm_kwargs)
        model.fit(X_train,verbose=verbose)

        # Evaluate
        train_ll = model.score(X_train)/len(X_train) # Normalized
        test_ll = model.score(X_test)/len(X_test)
        # Get test posteriors
        test_gamma, _ = model.predct_proba(X_test)
        test_states, _ = model.decode(X_test)
        results.append({
            'train_days':(train_day_start,train_day_end-1),
            'test_day':test_day,
            'train_ll':train_ll,
            'test_ll':test_ll,
            'test_gamma':test_gamma,
            'test_states':test_states,
            'model':model
        })
    if verbose:
        train_lls = [r['train_ll'] for r in results]
        test_lls = [r['test_ll'] for r in results]
        print(f"\nBacktest complete:{len(results)} windows")
        print(f"Train LL/sample:{np.mean(train_lls):.4f}± {np.std(train_lls):.4f}")
        print(f"Test LL/sample:{np.mean(test_lls):.4f}± {np.std(test_lls):.4f}")
    return results

# Model selection: compare across K states
def select_n_states(X, state_range=(2, 3, 4, 5), verbose=True, **hmm_kwargs):
    """
    Compare models with different numbers of states via BIC/AIC.
 
    Parameters
    ----------
    X : ndarray (T,) or (T, D)
    state_range : tuple of int
    **hmm_kwargs : passed to HMMGamma
 
    Returns
    -------
    results : dict mapping K -> {'model', 'bic', 'aic', 'll'}
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, np.newaxis]
    D = X.shape[1]
 
    results = {}
    for K in state_range:
        if verbose:
            print(f"Fitting K={K} states...")
        model = HMMGamma(n_states=K, n_features=D, **hmm_kwargs)
        model.fit(X, verbose=False)
        bic_val = model.bic(X)
        aic_val = model.aic(X)
        ll_val = model.score(X)
 
        results[K] = {
            'model': model,
            'bic': bic_val,
            'aic': aic_val,
            'll': ll_val
        }
        if verbose:
            print(f"  K={K}: BIC={bic_val:.1f}, AIC={aic_val:.1f}, LL={ll_val:.1f}")
 
    best_k = min(results, key=lambda k: results[k]['bic'])
    if verbose:
        print(f"\nBest K by BIC: {best_k}")
 
    return results




# Helper function
def _logsumexp(x):
    """Numerically stable log-sum-exp for 1D array"""
    x=np.asarray(x)
    c=x.max()
    if np.isinf(c):
        return -np.inf
    return c+np.log(np.sum(np.exp(x-c)))

def _logsumexp_axis(x,axis=0,keepdims=False):
    """Numerically stable log-sum-exp along an axis"""
    c = x.max(axis=axis,keepdims=True)
    out = c+np.log(np.sum(np.exp(x-c),axis=axis,keepdims=True))
    if not keepdims:
        out = np.squeeze(out,axis=axis)
    return out

def _count_runs(states,target_state = 0):
    """Count lengths of consecutive runs of target states"""
    runs = []
    count = 0
    for s in states:
        if s == target_state:
            count += 1
        else:
            if count > 0:
                runs.append(count)
            count = 0
    if count>0:
        runs.append(count)
    return runs
        