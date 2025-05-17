import numpy as np
import ot
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

offset = 0.5
seed = 42
centers = np.array([
    [0, 0],
    [offset, 0],
])
X, c = make_blobs(n_samples=50, centers=centers, n_features=2,
                  random_state=seed, cluster_std=0.05, shuffle=False)

X_source = X[c == 0]  # Source distribution
Y_target = X[c == 1]  # Target distribution

# (a)
# calculate
cost_matrix_l2 = ot.dist(X_source, Y_target, metric='euclidean')
cost_matrix_l2_squared = ot.dist(X_source, Y_target, metric='sqeuclidean')

# print
print("Euclidean distance matrix:")
print(cost_matrix_l2)

print("\nSquared Euclidean distance matrix:")
print(cost_matrix_l2_squared)

plt.imshow(cost_matrix_l2, cmap='viridis')
plt.title("Cost Matrix (Euclidean)")
plt.colorbar(label='Distance')
plt.show()

# 假设 X_source 和 Y_target 是前面定义的 25 个点

# (b)
n_source = X_source.shape[0]
n_target = Y_target.shape[0]
weights_source = np.ones(n_source) / n_source
weights_target = np.ones(n_target) / n_target

# 显示前几个权重确认
print("Source weights:", weights_source[:5])
print("Target weights:", weights_target[:5])

# 导入 EMD 函数
import ot

# (c)
transport_plan_l2 = ot.emd(weights_source, weights_target, cost_matrix_l2)
transport_plan_l2_squared = ot.emd(weights_source, weights_target, cost_matrix_l2_squared)

# 可选：查看部分结果
print("Transport Plan using Euclidean distance:")
print(transport_plan_l2[:5, :5])  # 只看前 5x5 的部分

print("\nTransport Plan using Squared Euclidean distance:")
print(transport_plan_l2_squared[:5, :5])

# (d)
def plot2D_samples_mat(xs, xt, G, ax, thr=1e-8, **kwargs):
    if ('color' not in kwargs) and ('c' not in kwargs):
        kwargs['color'] = 'k'  # default to black lines

    mx = G.max()
    if 'alpha' in kwargs:
        scale = kwargs['alpha']
        del kwargs['alpha']
    else:
        scale = 1

    for i in range(xs.shape[0]):
        for j in range(xt.shape[0]):
            if G[i, j] / mx > thr:
                ax.plot([xs[i, 0], xt[j, 0]],
                        [xs[i, 1], xt[j, 1]],
                        alpha=G[i, j] / mx * scale, **kwargs)

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# G1
axes[0].scatter(X_source[:, 0], X_source[:, 1], c='red', label='Source')
axes[0].scatter(Y_target[:, 0], Y_target[:, 1], c='blue', label='Target')
plot2D_samples_mat(X_source, Y_target, transport_plan_l2, axes[0], alpha=0.8)
axes[0].set_title("OT Plan (Euclidean)")
axes[0].legend()

# G2
axes[1].scatter(X_source[:, 0], X_source[:, 1], c='red', label='Source')
axes[1].scatter(Y_target[:, 0], Y_target[:, 1], c='blue', label='Target')
plot2D_samples_mat(X_source, Y_target, transport_plan_l2_squared, axes[1], alpha=0.8)
axes[1].set_title("OT Plan (Squared Euclidean)")
axes[1].legend()

plt.tight_layout()
plt.show()

# (f)
wasserstein_l2 = ot.emd2(weights_source, weights_target, cost_matrix_l2)
wasserstein_l2_squared = ot.emd2(weights_source, weights_target, cost_matrix_l2_squared)

print(f"Wasserstein distance (Euclidean): {wasserstein_l2:.4f}")
print(f"Wasserstein distance (Squared Euclidean): {wasserstein_l2_squared:.4f}")

