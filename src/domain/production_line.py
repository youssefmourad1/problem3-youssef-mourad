import numpy as np
import math
from src.config import ProductionLineConfig, Config
from src.utils.logger import setup_logger

logger = setup_logger("ProductionLineDomain")

class ProductionLine:
    def __init__(self, config=ProductionLineConfig):
        self.config = config
        self.n = config.N
        self.d = config.D
        self.c_p = config.C_P
        self.c_I = config.C_I 
        self.a_n_des = config.A_N_DES
        self.beta = config.BETA
        self.k = config.K
        self.p = config.P
        self.r = config.R
        self.m = config.M

    def calculate_rtilde_from_coupling_P(self, P_sub):
        r_tilde = np.zeros(self.n)
        r_tilde[0] = self.r[0] 

        for i in range(self.n - 1):
            P_val = P_sub[i]
            r_phys_next = self.r[i+1]
            p_phys_next = self.p[i+1]
            r_tilde_prev = r_tilde[i]
            
            numerator = P_val * p_phys_next * r_phys_next + (1 - P_val) * r_phys_next * r_tilde_prev
            denominator = P_val * p_phys_next + (1 - P_val) * r_phys_next
            r_tilde[i+1] = numerator / (denominator if denominator != 0 else 1e-12)
            
        return r_tilde[1:]
        
        # 1. POSITIONAL DECODING:
        # The PSO particle contains priority values for l_sub. We round these to binary 
        # to identify which machines have an inspection station.
        lamb_sub_rounded = np.round(lambda_sub)
        
        # Physical Constraint: Total stations must match the configuration (M-1 stations + 1 at sink)
        if abs(np.sum(lamb_sub_rounded) - self.m + 1) > 0.1:
             return Config.PENALTY_CONST, {"violations": [f"Lambda Sum {np.sum(lamb_sub_rounded)} != {self.m-1}"]}

        # Reconstruction: Pad the virtual vectors to N dimensions
        r_tilde = np.zeros(self.n); r_tilde[0] = self.r[0]; r_tilde[1:] = r_tilde_sub
        lamb = np.zeros(self.n); lamb[0:self.n-1] = lamb_sub_rounded; lamb[self.n-1] = 1.0 
        
        # 2. CONFORMABILITY LOGIC (q_i):
        # Calculate the ratio of bad parts in each buffer.
        # If an inspection station (lambda=1) is present, q resets for the next machine.
        q = np.zeros(self.n)
        q[0] = self.beta[0]
        for i in range(1, self.n):
            q[i] = (1 - lamb[i-1]) * q[i-1] * (1 + self.beta[i]) + self.beta[i]
            if q[i] >= 1.0 or q[i] < 0: return Config.PENALTY_CONST, {"violations": [f"q[{i}] invalid"]}

        # 3. DEMAND PROPAGATION (d_tilde):
        # The demand at machine i must account for all future conformability removals.
        d_tilde = np.zeros(self.n)
        for i in range(self.n):
            prod_val = 1.0
            for j in range(i, self.n): prod_val *= (1 + lamb[j] * q[j])
            d_tilde[i] = self.d * prod_val
        
        p_tilde = np.zeros(self.n)
        p_tilde[0] = self.p[0] 
        for i in range(1, self.n):
            denom = r_tilde[i] - r_tilde[i-1]
            if abs(denom) < 1e-12: return Config.PENALTY_CONST, {"violations": [f"r_tilde[{i}] approx r_tilde[{i-1}]"]}
            bracket = (self.p[i] / self.r[i]) + (self.p[i] * (self.r[i] - r_tilde[i]) * (self.r[i] + self.p[i])) / ((self.r[i]**2) * denom)
            p_tilde[i] = r_tilde[i] * bracket

        alpha = np.zeros(self.n - 1)
        for i in range(self.n - 1):
            alpha[i] = self.r[i+1]*(r_tilde[i+1] - r_tilde[i]) + self.p[i+1]*(self.r[i+1] - r_tilde[i+1])
            if abs(alpha[i]) < 1e-12: alpha[i] = 1e-12 
                
        P_vals = []; violations = []
        for i in range(self.n - 1):
             P_i = (self.r[i+1]*(r_tilde[i+1] - r_tilde[i])) / alpha[i]
             P_vals.append(P_i)
             
             term2 = ((self.r[i] + self.p[i]) / (self.r[i] * self.k[i])) * self.d
             if i == 0: lb = term2
             else:
                 prod_j = 1.0
                 for j in range(i): prod_j *= (self.r[j] / (self.r[j] + self.p[j]))
                 lb = max(prod_j, term2)
             
             prod_k = 1.0
             for j in range(i, self.n): prod_k *= ((self.r[j] + self.p[j]) / self.r[j])
             ub = min(prod_k * self.a_n_des, 1.0)
             
             if P_i < lb - 1e-4 or P_i > ub + 1e-4:
                 violations.append(f"P[{i}]={P_i:.4f} outside bounds")

        sum_T_i = 0.0
        for i in range(self.n - 1):
            try:
                den_A = (p_tilde[i] + r_tilde[i])**2
                term_A1 = (self.k[i] * p_tilde[i]) / den_A
                num_A2 = self.k[i] * r_tilde[i] * self.r[i+1] * (r_tilde[i] + p_tilde[i]) * (r_tilde[i+1] - r_tilde[i])
                A_i = term_A1 - num_A2 / (d_tilde[i] * alpha[i])
                
                num_B = self.k[i] * self.p[i+1] * (self.r[i+1] - r_tilde[i+1])
                den_B_inner = (self.k[i] * r_tilde[i] * self.r[i+1] * (r_tilde[i+1] - r_tilde[i])) / (d_tilde[i])
                B_i = num_B / ((r_tilde[i] + p_tilde[i]) * alpha[i] - den_B_inner)
                
                term_C_num1 = (self.k[i] - d_tilde[i]) * self.r[i+1] * (r_tilde[i+1] - r_tilde[i]) - d_tilde[i] * self.p[i+1] * (self.r[i+1] - r_tilde[i+1])
                term_C_num2 = d_tilde[i] * (p_tilde[i] + r_tilde[i]) - self.k[i] * r_tilde[i]
                num_C = d_tilde[i] * alpha[i] * term_C_num1 * term_C_num2
                den_C = d_tilde[i] * (p_tilde[i] + r_tilde[i]) * alpha[i] - self.k[i] * r_tilde[i] * self.r[i+1] * (r_tilde[i+1] - r_tilde[i])
                C_i = num_C / den_C
                
                D_i = (p_tilde[i] * d_tilde[i]) / (r_tilde[i] * (self.k[i] * (self.r[i+1] * (r_tilde[i+1] - r_tilde[i])) / alpha[i] - d_tilde[i]))
                den_E_sing = (self.r[i+1] - r_tilde[i+1])
                den_E_bracket = self.k[i] * self.r[i+1] * (r_tilde[i+1] - r_tilde[i]) - d_tilde[i] * alpha[i]
                den_E = r_tilde[i] * (p_tilde[i] + r_tilde[i]) * self.p[i+1] * den_E_sing * den_E_bracket
                E_i = (alpha[i] * p_tilde[i] * (alpha[i] * (p_tilde[i] + r_tilde[i]) * d_tilde[i] - self.k[i] * r_tilde[i] * self.r[i+1] * (r_tilde[i+1] - r_tilde[i]))) / den_E
                
                diff = D_i - E_i
                if diff <= 1e-9:
                     return Config.PENALTY_CONST, {"violations": [f"Log Rejection machine {i+1}"]}
                
                T_i = A_i - B_i - C_i * math.log(diff)
                sum_T_i += T_i
                
            except (ZeroDivisionError, ValueError, OverflowError):
                return Config.PENALTY_CONST, {"violations": [f"Math Error machine {i+1}"]}

        idx_n = self.n - 1; r_tn = r_tilde[idx_n]; p_tn = p_tilde[idx_n]; k_n = self.k[idx_n]; d_tn = d_tilde[idx_n]
        try:
             rho_n = (r_tn * (k_n - d_tn / self.a_n_des)) / (p_tn * (d_tn / self.a_n_des))
             if rho_n >= 1.0 or rho_n <= 0: return Config.PENALTY_CONST, {"violations": ["rho_n invalid"]}
                 
             mu_n = p_tn / (k_n - d_tn / self.a_n_des)
             term1 = self.c_p * k_n * (self.a_n_des * (p_tn + r_tn) - r_tn) / ((p_tn + r_tn) * (1 - rho_n) * p_tn)
             fraction = (1 - rho_n) / ((1 - self.a_n_des) * ((p_tn + r_tn) / p_tn))
             inner_log = (1.0 / rho_n) * (1 - fraction)
             
             if inner_log <= 1e-9:
                  return Config.PENALTY_CONST, {"violations": ["Log Rejection T_n"]}
                  
             T_n = term1 + ((p_tn + r_tn) * (1 - self.a_n_des) / (mu_n * ((1 - rho_n)**2) * p_tn) - 1.0 / (mu_n * (1 - rho_n))) * math.log(inner_log)
        except (ZeroDivisionError, ValueError, OverflowError):
            return Config.PENALTY_CONST, {"violations": ["Math Error T_n"]}
            
        machine_cost = sum_T_i + T_n
        inspection_cost = self.c_I * np.sum(lamb * d_tilde)
        total_cost = machine_cost + inspection_cost
        
        # Singularity protection
        if total_cost < -50:
             return Config.PENALTY_CONST, {"violations": ["Analytical Singularity Detected"]}
            
        return total_cost, {
            "r_tilde": r_tilde,
            "lambda": lamb,
            "P_vals": P_vals,
            "violations": violations,
            "Cost": total_cost,
            "machine_cost": machine_cost,
            "inspection_cost": inspection_cost
        }
