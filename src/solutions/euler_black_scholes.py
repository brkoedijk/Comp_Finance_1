import numpy as np
from scipy.stats import norm

class EulerBlackScholes:
    def __init__(self, S0, r, sigma_true, sigma_model, T, N):
        """
        Initialize with separate volatilities for simulation and model.
        
        Parameters:
        S0 : Initial stock price
        r : Risk-free rate
        sigma_true : Actual volatility for stock price simulation
        sigma_model : Volatility used in option pricing and delta calculation
        T : Time to maturity
        N : Number of time steps
        """
        self.S0 = S0
        self.r = r                  # Risk-free rate
        self.sigma_true = sigma_true  # True volatility for simulation
        self.sigma_model = sigma_model  # Model volatility for pricing/hedging
        self.T = T                  # Time to maturity
        self.N = N                  # Number of time steps
        self.dt = T / N             # Time step size

    def call_price(self, S, K, tau):
        """Calculate Black-Scholes price for a European call option using model volatility."""
        if tau <= 0:
            return max(0, S - K)
        d1 = (np.log(S / K) + (self.r + 0.5 * self.sigma_model**2) * tau) / (self.sigma_model * np.sqrt(tau))
        d2 = d1 - self.sigma_model * np.sqrt(tau)
        return S * norm.cdf(d1) - K * np.exp(-self.r * tau) * norm.cdf(d2)

    def call_delta(self, S, K, tau):
        """Calculate Black-Scholes delta for a European call option using model volatility."""
        if tau <= 0:
            return 1.0 if S > K else 0.0
        d1 = (np.log(S / K) + (self.r + 0.5 * self.sigma_model**2) * tau) / (self.sigma_model * np.sqrt(tau))
        return norm.cdf(d1)

    def simulate(self):
        """Simulate stock price path using Euler discretization with true volatility."""
        time = np.linspace(0, self.T, self.N + 1)
        S = np.zeros(self.N + 1)
        S[0] = self.S0

        for t in range(1, len(S)):
            dw = np.random.normal(0, np.sqrt(self.dt))  # Brownian motion increment
            # Euler-Maruyama scheme for geometric Brownian motion
            S[t] = S[t-1] * np.exp((self.r - 0.5 * self.sigma_true**2) * self.dt + self.sigma_true * dw)
        
        return time, S
    
    def delta_hedge_short_call(self, K, hedge_interval):
        """
        Simulate delta hedging of a short call option position with volatility mismatch.
        
        Parameters:
        K : Strike price
        hedge_interval : Number of time steps between hedging adjustments
        
        Returns:
        dict: Results of the hedging simulation
        """
        # Simulate stock price path with true volatility
        time, S = self.simulate()
        
        # Initial option price and delta using model volatility
        option_price = self.call_price(self.S0, K, self.T)
        initial_delta = self.call_delta(self.S0, K, self.T)
        
        # Initialize hedging portfolio
        cash_account = option_price  # Received premium for short call
        stock_position = initial_delta  # Long delta shares to hedge
        cash_account -= initial_delta * self.S0  # Cost to buy initial shares
        
        # Tracking variables
        portfolio_values = [cash_account + stock_position * self.S0]
        option_values = [option_price]
        delta_values = [initial_delta]
        hedge_times = [0]
        
        # Perform hedging at specified intervals
        for i in range(hedge_interval, self.N+1, hedge_interval):
            current_time = time[i]
            time_to_maturity = self.T - current_time
            
            # Accrue interest on cash account
            cash_account *= np.exp(self.r * hedge_interval * self.dt)
            
            # Current option value and delta using model volatility
            current_option = self.call_price(S[i], K, time_to_maturity)
            current_delta = self.call_delta(S[i], K, time_to_maturity)
            
            # Adjust hedge position
            shares_to_trade = current_delta - stock_position
            cash_account -= shares_to_trade * S[i]
            stock_position = current_delta
            
            # Record values
            portfolio_value = cash_account + stock_position * S[i]
            portfolio_values.append(portfolio_value)
            option_values.append(current_option)
            delta_values.append(current_delta)
            hedge_times.append(current_time)
        
        # Terminal settlement
        option_payoff = max(0, S[self.N] - K)  # Call option payoff at maturity
        
        # Close out stock position
        cash_account += stock_position * S[self.N]
        stock_position = 0
        
        # Pay option payoff (we're short the option)
        cash_account -= option_payoff
        
        # Final P&L
        final_pnl = cash_account
        
        # Add final values to tracking arrays
        portfolio_values.append(cash_account)
        option_values.append(option_payoff)
        hedge_times.append(self.T)
        
        return {
            'final_pnl': final_pnl,
            'portfolio_values': np.array(portfolio_values),
            'option_values': np.array(option_values),
            'delta_values': np.array(delta_values),
            'hedge_times': np.array(hedge_times),
            'stock_path': S,
            'time': time
        }