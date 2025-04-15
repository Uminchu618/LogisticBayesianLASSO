import numpy as np
import pandas as pd
from polyagamma import random_polyagamma
import concurrent.futures
from tqdm import tqdm


def generate_logistic_data(n, p, beta_true):
    # サンプルデータの生成
    X = np.random.randn(n, p)
    linear_pred = X @ beta_true
    p_i = 1.0 / (1.0 + np.exp(-linear_pred))
    y = np.random.binomial(1, p_i)
    return X, y


def gibbs_sampler_logistic_pg(X, y, n_iter=2000, lambda1=1.0):
    # ポリアガンマ分布を用いたロジスティック回帰のギブスサンプリング
    n, p = X.shape
    # バイアス項を追加
    X = np.hstack([np.ones((n, 1)), X])
    p = p + 1  # 新たな次元数
    beta = np.zeros(p)
    samples = np.zeros((n_iter, p))
    for t in range(n_iter):
        ## ロジスティック尤度の寄与
        omega = random_polyagamma(1, z=(X @ beta), size=n)
        kappa = y - 0.5
        W = np.diag(omega)
        Q = X.T @ W @ X
        r = X.T @ kappa
        # Lasso項の寄与：対角行列 D_t = diag(1/τ_{t,j}^2)
        if lambda1 > 0.0001:
            tau2 = np.random.exponential(scale=2.0 / (lambda1**2), size=p)
            D = np.diag(1.0 / tau2)
        else:
            D = np.zeros((p, p))
        sigma = np.linalg.inv(Q + D)
        mu = sigma @ r
        # β の事後分布
        beta = np.random.multivariate_normal(mu, sigma)
        samples[t] = beta
    return samples


def run_simulation(seed):
    # 各プロセスでシードを固定して再現性を確保
    np.random.seed(seed)
    # 真のパラメータ
    beta_true = np.array([1.0, -2.0, 0.1])
    X, y = generate_logistic_data(n=500, p=3, beta_true=beta_true)

    np.random.seed(seed)

    samples = gibbs_sampler_logistic_pg(X, y, n_iter=3000)
    burn_in = 1000
    posterior_samples = samples[burn_in:]
    beta_est = posterior_samples.mean(axis=0)
    return beta_est


def run_simulation_lambda(args):
    # lambda1 の値を変更してシミュレーションを実行する関数
    seed, lambda1 = args
    np.random.seed(seed)
    beta_true = np.array([1.0, -2.0, 0.1])
    X, y = generate_logistic_data(n=1000, p=3, beta_true=beta_true)
    samples = gibbs_sampler_logistic_pg(X, y, n_iter=3000, lambda1=lambda1)
    burn_in = 1000
    posterior_samples = samples[burn_in:]
    beta_est = posterior_samples.mean(axis=0)
    return beta_est


def run_lasso_path_simulation():
    # 複数の lambda1 (Lasso の正則化パラメータ) についてシミュレーションを実行し、各回帰係数の推定値を集計
    lambda_grid = np.logspace(-2, 2, num=10)
    n_runs = 100  # 各 lambda1 に対するシミュレーション回数
    results = []
    for lam in lambda_grid:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            seeds = [(seed, lam) for seed in range(n_runs)]
            beta_estimates = list(
                tqdm(
                    executor.map(run_simulation_lambda, seeds),
                    total=n_runs,
                    desc=f"Lambda={lam:0.4f}",
                )
            )
        beta_estimates = np.array(beta_estimates)
        beta_mean = beta_estimates.mean(axis=0)
        results.append([lam] + beta_mean.tolist())
    df = pd.DataFrame(results, columns=["lambda1", "beta0", "beta1", "beta2", "beta3"])
    df.to_csv("lasso_path_estimates.csv", index=False)
    print(
        "Lasso Path シミュレーションが完了しました。lasso_path_estimates.csv に結果が保存されました。"
    )


if __name__ == "__main__":
    # 既存のシミュレーション
    n_runs = 1000  # シミュレーション回数
    with concurrent.futures.ProcessPoolExecutor() as executor:
        beta_estimates = list(
            tqdm(
                executor.map(run_simulation, range(n_runs)),
                total=n_runs,
                desc="シミュレーション進行中",
            )
        )
    beta_estimates = np.array(beta_estimates)
    df = pd.DataFrame(beta_estimates, columns=["beta0", "beta1", "beta2", "beta3"])
    df.to_csv("beta_estimates.csv", index=False)
    print("シミュレーションが完了しました。beta_estimates.csv に結果が保存されました。")

    # 新たに追加した Lasso Path シミュレーション
    run_lasso_path_simulation()
