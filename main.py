import numpy as np
import matplotlib.pyplot as plt
from src.domain.production_line import ProductionLine
from src.optimizer.pso import PSO
from src.config import ProductionLineConfig, Config
from src.utils.logger import setup_logger
from src.visualization.plotter import plot_convergence
import os

logger = setup_logger("Main")

def main():
    logger.info("Starting PSO Optimization for Unreliable Production Line")

    # 1. Initialize Domain
    line_model = ProductionLine(ProductionLineConfig)
    n = ProductionLineConfig.N
    
    # 2. Problem Dimensions
    dim_r = n - 1
    dim_l = n - 1
    
    # 3. Setup PSO
    bounds = {
        'r_min': 0.1,
        'r_max': 0.99
    }

    # 4. Objective Wrapper
    def objective(pos):
        P_sub = pos[:dim_r]
        l_sub = pos[dim_r:]
        r_tilde_sub = line_model.calculate_rtilde_from_coupling_P(P_sub)
        cost, _ = line_model.calculate_cost(r_tilde_sub, l_sub)
        return cost

    # 5. Initialization
    initial_P = np.array([0.9252, 0.9918, 0.6917, 0.9546, 0.7908, 0.6083, 0.6829, 0.7576, 0.5845])
    initial_L = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    initial_guess = np.concatenate([initial_P, initial_L])
    
    # 6. Run PSO
    pso = PSO(objective, dim_r, dim_l, bounds, initial_guess=initial_guess)
    best_pos, best_cost, history = pso.optimize()

    # 7. Final Report
    logger.info("Optimization Complete.")
    
    if best_cost < Config.PENALTY_CONST:
        logger.info(f"Best Cost Found: {best_cost:.6f}")
        
        P_sol = best_pos[:dim_r]
        l_sol = best_pos[dim_r:]
        r_sol = line_model.calculate_rtilde_from_coupling_P(P_sol)
        _, details = line_model.calculate_cost(r_sol, l_sol)
        
        logger.info("--- Best Solution Details ---")
        if 'violations' in details and details['violations']:
            logger.info("Status Summary:")
            for v in details['violations']:
                logger.info(f"  - {v}")
                
        if 'P_vals' in details:
             logger.info(f"P_vals: {[round(p, 4) for p in details['P_vals']]}")
             
        if 'r_tilde' in details:
            logger.info(f"r_tilde (full virtual): {details.get('r_tilde')}")
            logger.info(f"lambda (inspection stations): {details.get('lambda')}")
    else:
        logger.error("Failed to find a strictly feasible solution.")
        logger.info(f"Lowest Penalty Value: {best_cost:.6f}")

    # 8. Visualization
    if not os.path.exists('results'):
        os.makedirs('results')
    plot_convergence(history, "results/convergence.png")
    logger.info("Convergence plot saved to results/convergence.png")

if __name__ == "__main__":
    main()
