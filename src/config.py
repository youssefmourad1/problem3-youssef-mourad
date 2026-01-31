import numpy as np

class Config:
    # PSO Parameters
    SWARM_SIZE = 80
    MAX_ITER = 300
    W = 0.7  # Inertia weight
    C1 = 1.5 # Cognitive coefficient
    C2 = 1.5 # Social coefficient
    
    # Initialization
    STAIRCASE_FACTOR = 0.01 
    
    # Problem Constraints (P-Space Search)
    R_TILDE_MIN = 0.1  
    R_TILDE_MAX = 0.9   
    
    # Penalties
    PENALTY_CONST = 1e18 # Death Penalty (Strict Rejection)
    CONSTRAINT_PENALTY = 1e12 # Gradient for bounds
    
    # Random Seed
    SEED = 42

class ProductionLineConfig:
    # Table 2 Data
    N = 10
    D = 1.0
    C_P = 1.0
    C_I = 2.0 
    A_N_DES = 0.95
    M = 4 
    
    # Table 1 Data
    BETA = np.array([0.1]*10)
    K = np.array([4.0]*10)
    P = np.array([0.1, 0.2, 0.3, 0.15, 0.1, 0.25, 0.05, 0.27, 0.12, 0.18])
    R = np.array([0.8, 0.6, 0.7, 0.9, 0.85, 0.65, 0.75, 0.95, 0.85, 0.92])
    
    @staticmethod
    def get_parameter_dict():
        return {
            "N": ProductionLineConfig.N,
            "D": ProductionLineConfig.D,
            "C_P": ProductionLineConfig.C_P,
            "C_I": ProductionLineConfig.C_I,
            "A_N_DES": ProductionLineConfig.A_N_DES,
            "M": ProductionLineConfig.M,
            "BETA": ProductionLineConfig.BETA,
            "K": ProductionLineConfig.K,
            "P": ProductionLineConfig.P,
            "R": ProductionLineConfig.R
        }
