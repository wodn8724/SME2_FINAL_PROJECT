import numpy as np
import scipy.io as sio
from scipy.optimize import least_squares

def your_algorithm(d_u, p_bs):
    d = np.asarray(d_u, dtype=float).reshape(-1)
    
    min_x, max_x = np.min(p_bs[0]), np.max(p_bs[0])
    min_y, max_y = np.min(p_bs[1]), np.max(p_bs[1])
    grid_res = 2.0
    x_c = np.arange(min_x - 10, max_x + 10, grid_res)
    y_c = np.arange(min_y - 10, max_y + 10, grid_res)
    X_c, Y_c = np.meshgrid(x_c, y_c)
    grid = np.vstack([X_c.ravel(), Y_c.ravel()]).T
    
    diff_c = grid[:, np.newaxis, :] - p_bs.T[np.newaxis, :, :]
    dist_mat = np.sqrt(np.sum(diff_c**2, axis=2))
    loss = np.sum(np.sort((d - dist_mat)**2, axis=1)[:, :8], axis=1)
    pos = grid[np.argmin(loss)]

    for _ in range(4):
        diff = p_bs - pos.reshape(2, 1)
        pred_d = np.sqrt(np.sum(diff**2, axis=0))
        residual = d - pred_d
        
        weights = 1.0 / (1.0 + (np.clip(residual, 0, 5) / 1.0)**2)
        
        def func(x):
            diff = p_bs - x.reshape(2, 1)
            return weights * (np.sqrt(np.sum(diff**2, axis=0)) - d)
            
        res = least_squares(func, pos, loss='linear')
        pos = res.x
        
    return pos

def main():
    mat_path = 'DH_FR1.mat'
    data = sio.loadmat(mat_path, squeeze_me=False)
    BS_positions = np.asarray(data['BS_positions'], dtype=float)
    d_hat = np.asarray(data['d_hat'], dtype=float)

    num_user = d_hat.shape[1]
    p_hat = np.zeros((2, num_user))
    
    for u in range(num_user):
        p_hat[:, u] = your_algorithm(d_hat[:, u], BS_positions)

    return p_hat

if __name__ == "__main__":
    main()
