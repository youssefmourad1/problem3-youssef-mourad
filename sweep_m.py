import numpy as np
import matplotlib.pyplot as plt
import time
import os
import pandas as pd
import seaborn as sns
from src.domain.production_line import ProductionLine
from src.optimizer.pso import PSO
from src.config import ProductionLineConfig, Config
from src.utils.logger import setup_logger
from src.visualization.plotter import plot_convergence

logger = setup_logger("SweepM")

def find_safe_seed_for_m(line, m_total, max_trials=10000):
    """Dynamically find a valid seed for any M."""
    lb_list = []; ub_list = []
    r = line.r; p = line.p; k = line.k; d = line.d; a_n_des = line.a_n_des
    for i in range(1, 10):
        idx = i - 1
        term2 = ((r[idx] + p[idx]) / (r[idx] * k[idx])) * d
        lb = term2 if i == 1 else max(1e-4, term2) # Simple bounds check
        
        prod_k = 1.0
        for j in range(i, 11): prod_k *= ((r[j-1] + p[j-1]) / r[j-1])
        ub = min(prod_k * a_n_des, 1.0)
        lb_list.append(lb); ub_list.append(ub)
    
    lb_arr = np.array(lb_list); ub_arr = np.array(ub_list)
    lamb_sub = np.zeros(9)
    if m_total > 1:
        ratios = (p/r)[:9]
        indices = np.argsort(ratios)[-(m_total-1):]
        lamb_sub[indices] = 1.0

    for _ in range(max_trials):
        P_try = np.random.uniform(lb_arr, ub_arr)
        r_sol = line.calculate_rtilde_from_coupling_P(P_try)
        cost, _ = line.calculate_cost(r_sol, lamb_sub)
        if cost < 1e15:
            return P_try, lamb_sub
    return None, None

def main():
    if not os.path.exists('results'):
        os.makedirs('results')

    results = []
    station_history = [] 
    best_config = None
    min_total_cost = float('inf')
    
    m_range = range(1, 10)
    
    for m in m_range:
        logger.info(f"--- Benchmarking M = {m} ---")
        ProductionLineConfig.M = m
        line = ProductionLine(ProductionLineConfig)
        
        safe_P, safe_L = find_safe_seed_for_m(line, m)
        if safe_P is None:
            logger.error(f"Failed to find seed for M={m}. Skipping.")
            continue
        
        initial_guess = np.concatenate([safe_P, safe_L])
        start_time = time.time()
        
        def objective(pos):
            P_sub = pos[:9]; l_sub = pos[9:]
            cost, _ = line.calculate_cost(line.calculate_rtilde_from_coupling_P(P_sub), l_sub)
            return cost

        pso = PSO(objective, 9, 9, {'r_min': 0.1, 'r_max': 0.99}, initial_guess=initial_guess)
        best_pos, best_cost, history = pso.optimize()
        exec_time = time.time() - start_time
        
        P_sol = best_pos[:9]; L_sol = best_pos[9:]
        r_sol = line.calculate_rtilde_from_coupling_P(P_sol)
        _, details = line.calculate_cost(r_sol, L_sol)
        
        lamb_final = details.get('lambda', np.zeros(10))
        station_history.append(lamb_final)
        
        m_cost = details.get('machine_cost', 0)
        i_cost = details.get('inspection_cost', 0)
        
        results.append({
            'M': m,
            'TotalCost': best_cost,
            'MachineCost': m_cost,
            'InspectionCost': i_cost,
            'Time': exec_time,
            'P_avg': np.mean(P_sol)
        })
        
        if best_cost < min_total_cost:
            min_total_cost = best_cost
            best_config = {'M': m, 'r_tilde': details.get('r_tilde'), 'lambda': lamb_final}
        
        plot_convergence(history, f"results/convergence_M{m}.png")
        logger.info(f"M={m} Complete. Cost={best_cost:.4f}")
        
    df = pd.DataFrame(results)
    df.to_csv('results/summary.csv', index=False)
    
    # --- PLOTTING ---
    # 1. Total Cost vs M
    plt.figure(figsize=(10, 6))
    plt.plot(df['M'], df['TotalCost'], 'o-', color='tab:blue', label='Total Cost', linewidth=2)
    plt.title("Production Line Cost vs M")
    plt.xlabel("Number of Inspection Stations (M)")
    plt.ylabel("Cost")
    plt.legend(); plt.grid(True)
    plt.savefig('results/cost_vs_m.png')
    
    # 2. Cost Breakdown vs M
    plt.figure(figsize=(10, 6))
    plt.bar(df['M'], df['MachineCost'], label='Machine Cost (Maintenance/Storage)', color='tab:orange', alpha=0.7)
    plt.bar(df['M'], df['InspectionCost'], bottom=df['MachineCost'], label='Inspection Cost', color='tab:green', alpha=0.7)
    plt.title("Cost Breakdown per M")
    plt.xlabel("M")
    plt.ylabel("Cost Components")
    plt.legend(); plt.grid(True, axis='y')
    plt.savefig('results/cost_breakdown.png')
    
    # 3. Station Heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(np.array(station_history), annot=True, cmap="YlGnBu", 
                xticklabels=[f"M{i+1}" for i in range(10)],
                yticklabels=[f"M={m}" for m in m_range])
    plt.title("Inspection Station Placement Strategy")
    plt.savefig('results/station_heatmap.png')
    
    # 4. Global Best r_tilde Profile
    if best_config:
        plt.figure(figsize=(10, 6))
        machines = np.arange(1, 11)
        plt.plot(machines, best_config['r_tilde'], 'D-', color='tab:red', label=f'Best M={best_config["M"]}')
        plt.title(f"Optimal Virtual Repair Rate (r_tilde) Profile @ M={best_config['M']}")
        plt.xlabel("Machine Index")
        plt.ylabel("Virtual Repair Rate")
        plt.xticks(machines); plt.grid(True); plt.legend()
        plt.savefig('results/optimal_rtilde_profile.png')

    logger.info("M-Sweep Complete. All expanded plots saved to results/")

if __name__ == "__main__":
    main()
