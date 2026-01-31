import numpy as np
from src.config import Config, ProductionLineConfig
from src.utils.logger import setup_logger

logger = setup_logger("PSO_Optimizer")

class PSO:
    def __init__(self, objective_function, dim_r, dim_l, bounds, initial_guess=None,
                 w=Config.W, c1=Config.C1, c2=Config.C2, staircase_factor=Config.STAIRCASE_FACTOR):
        """
        Initializes the PSO optimizer.
        """
        self.func = objective_function
        self.dim_r = dim_r
        self.dim_l = dim_l
        self.dim = dim_r + dim_l
        self.bounds = bounds
        self.initial_guess = initial_guess
        
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.staircase_factor = staircase_factor
        
        self.swarm_size = Config.SWARM_SIZE
        self.max_iter = Config.MAX_ITER
        self.M = ProductionLineConfig.M - 1
        
        self.positions = []
        self.velocities = []
        self.pbest_pos = []
        self.pbest_val = []
        
        self.gbest_pos = None
        self.gbest_val = float('inf')
        self.history = []
        
        self._initialize_swarm()

    def _decode_position(self, pos):
        """
        PRIORITY ENCODING DECODER:
        Converts the hidden continuous particle space into the physical problem space.
        1. Clips r_tilde within feasibility bounds defined in config.
        2. Applies Rank-Based Priority Mapping for Lambda:
           The M-top continuous values in the lambda part of the particle are selected 
           as '1' (inspection station present), others are '0'. 
           This ensures we always satisfy the M-station constraint.
        """
        decoded = pos.copy()
        
        # Continuous r_part
        r_part = decoded[:self.dim_r]
        r_part = np.clip(r_part, self.bounds['r_min'], self.bounds['r_max'])
        decoded[:self.dim_r] = r_part
        
        # Binary-mapped l_part
        l_part_continuous = decoded[self.dim_r:]
        l_part_binary = np.zeros_like(l_part_continuous)
        
        if self.M > 0 and self.M <= self.dim_l:
            top_m_indices = np.argsort(l_part_continuous)[-self.M:]
            l_part_binary[top_m_indices] = 1.0
            
        decoded[self.dim_r:] = l_part_binary
        return decoded

    def _staircase_initialization(self):
        full_r = ProductionLineConfig.R
        r_tilde_sim = np.zeros(len(full_r))
        r_tilde_sim[0] = full_r[0] 
        
        for i in range(1, len(full_r)):
            factor = 0.99
            val = full_r[i] * factor
            noise = np.random.uniform(-0.001, 0.001)
            val += noise
            if val >= full_r[i] - 1e-4:
                val = full_r[i] - 0.001
            r_tilde_sim[i] = val
            
        return r_tilde_sim[1:]

    def _initialize_swarm(self):
        """
        HYBRID INITIALIZATION STRATEGY:
        1. Validated Guess (Particle 0): If an initial_guess is provided (e.g., the 'Safe Seed'),
           it is used as the first particle to bootstrap convergence.
        2. Gaussian Diversification: Other particles are initialized near the guess with 
           controlled noise to explore the feasibility neighborhood.
        3. Staircase Fallback: If no guess is provided, a descending staircase initialization
           is used as a heuristic for the virtual repair rate profile.
        """
        logger.info(f"Initializing swarm with {self.swarm_size} particles...")
        valid_count = 0
        
        for i in range(self.swarm_size):
            if self.initial_guess is not None:
                if len(self.initial_guess) == self.dim:
                    bounds_width = self.bounds['r_max'] - self.bounds['r_min']
                    noise_r = np.random.normal(0, 0.02 * bounds_width, self.dim_r)
                    r_part = self.initial_guess[:self.dim_r] + noise_r
                    r_part = np.clip(r_part, self.bounds['r_min'], self.bounds['r_max'])
                    
                    l_part = self.initial_guess[self.dim_r:] + np.random.normal(0, 0.05, self.dim_l)
                    l_part = np.clip(l_part, 0, 1)

                    if i == 0:
                        r_part = self.initial_guess[:self.dim_r].copy()
                        l_part = self.initial_guess[self.dim_r:].copy()
                else:
                    bounds_width = self.bounds['r_max'] - self.bounds['r_min']
                    noise = np.random.normal(0, 0.05 * bounds_width, self.dim_r)
                    r_part = self.initial_guess + noise
                    r_part = np.clip(r_part, self.bounds['r_min'], self.bounds['r_max'])
                    l_part = np.random.uniform(0, 1, self.dim_l)
                    
                    if i == 0:
                         r_part = self.initial_guess.copy()
                         l_part = np.zeros(self.dim_l)
                         l_part[-self.M:] = 1.0 
            else:
                r_part = self._staircase_initialization()
                l_part = np.random.uniform(0, 1, self.dim_l)
            
            pos = np.concatenate([r_part, l_part])
            decoded_pos = self._decode_position(pos)
            val = self.func(decoded_pos)
            
            if val < 1.001 * Config.PENALTY_CONST:
                valid_count += 1
            
            vel = np.random.uniform(-0.1, 0.1, self.dim)
            self.positions.append(pos)
            self.velocities.append(vel)
            self.pbest_pos.append(pos.copy())
            self.pbest_val.append(val)
            
            if val < self.gbest_val:
                self.gbest_val = val
                self.gbest_pos = pos.copy()
        
        logger.info(f"Initialization complete. Near-Valid Particles: {valid_count}/{self.swarm_size}")
        if self.gbest_val <= 1e15:
            logger.info(f"Best initial cost: {self.gbest_val}")

    def optimize(self):
        logger.info("Starting optimization loop...")
        for t in range(self.max_iter):
            valid_particles = 0
            for i in range(self.swarm_size):
                r1 = np.random.rand(self.dim); r2 = np.random.rand(self.dim)
                inertia = self.w * self.velocities[i]
                cognitive = self.c1 * r1 * (self.pbest_pos[i] - self.positions[i])
                social = self.c2 * r2 * (self.gbest_pos - self.positions[i])
                self.velocities[i] = inertia + cognitive + social
                self.positions[i] += self.velocities[i]
                
                # Clipping
                self.positions[i][:self.dim_r] = np.clip(self.positions[i][:self.dim_r], self.bounds['r_min'], self.bounds['r_max'])
                self.positions[i][self.dim_r:] = np.clip(self.positions[i][self.dim_r:], 0, 1)
                
                decoded_pos = self._decode_position(self.positions[i])
                val = self.func(decoded_pos)
                
                if val < self.pbest_val[i]:
                    self.pbest_val[i] = val
                    self.pbest_pos[i] = self.positions[i].copy()
                    if val < self.gbest_val:
                        self.gbest_val = val
                        self.gbest_pos = self.positions[i].copy()

                if val < Config.PENALTY_CONST:
                    valid_particles += 1

            self.history.append(self.gbest_val)
            if t % 10 == 0 or t == self.max_iter - 1:
                logger.info(f"Iteration {t}: Best Cost = {self.gbest_val:.4f} | Valid Particles: {valid_particles}/{self.swarm_size}")

        return self.gbest_pos, self.gbest_val, self.history
