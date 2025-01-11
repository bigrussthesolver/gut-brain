import numpy as np
from typing import List, Tuple, Dict

class GutNetwork:
    def __init__(self, state_size: int = 5):
        self.state_size = state_size
        self.internal_state = np.zeros(state_size)
        self.homeostasis_target = np.ones(state_size) * 0.5  # Target middle state
        self.adaptation_rate = 0.1
        self.stress_threshold = 0.7
        
    def process_input(self, input_signal: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Process input and update internal state
        Returns: processed signal and stress level
        """
        # Update internal state based on input
        delta = input_signal - self.internal_state
        self.internal_state += self.adaptation_rate * delta
        
        # Calculate stress level based on deviation from homeostasis
        stress_level = np.mean(np.abs(self.internal_state - self.homeostasis_target))
        
        # Generate gut response
        gut_response = self.generate_response(stress_level)
        
        return gut_response, stress_level
    
    def generate_response(self, stress_level: float) -> np.ndarray:
        """Generate response based on internal state and stress"""
        if stress_level > self.stress_threshold:
            # Strong deviation from homeostasis - defensive response
            response = np.clip(1.5 * self.internal_state, 0, 1)
        else:
            # Normal operation - maintain balance
            response = self.internal_state
            
        return response

class BrainNetwork:
    def __init__(self, input_size: int = 5):
        self.input_size = input_size
        self.memory = []
        self.confidence_threshold = 0.6
        
    def process_input(self, input_signal: np.ndarray, gut_signal: np.ndarray, 
                     stress_level: float) -> Tuple[np.ndarray, Dict]:
        """
        Process input considering both direct input and gut signals
        Returns: decision and metadata
        """
        # Combine input with gut signal
        combined_signal = self.integrate_signals(input_signal, gut_signal, stress_level)
        
        # Make decision
        decision = self.make_decision(combined_signal, stress_level)
        
        # Update memory
        self.update_memory(input_signal, gut_signal, decision)
        
        metadata = {
            'confidence': self.calculate_confidence(decision, stress_level),
            'gut_influence': stress_level
        }
        
        return decision, metadata
    
    def integrate_signals(self, input_signal: np.ndarray, gut_signal: np.ndarray, 
                         stress_level: float) -> np.ndarray:
        """Integrate direct input with gut signal"""
        # Higher stress means more weight to gut signal
        gut_weight = stress_level
        input_weight = 1 - stress_level
        
        return (input_weight * input_signal + gut_weight * gut_signal) / 2

    def make_decision(self, combined_signal: np.ndarray, stress_level: float) -> np.ndarray:
        """Make final decision based on combined signal"""
        if stress_level > self.confidence_threshold:
            # Under high stress, make more conservative decisions
            decision = np.clip(combined_signal * 0.8, 0, 1)
        else:
            decision = combined_signal
            
        return decision
    
    def update_memory(self, input_signal: np.ndarray, gut_signal: np.ndarray, 
                     decision: np.ndarray):
        """Store experience in memory"""
        experience = {
            'input': input_signal.copy(),
            'gut_signal': gut_signal.copy(),
            'decision': decision.copy()
        }
        self.memory.append(experience)
        
        # Keep memory size manageable
        if len(self.memory) > 100:
            self.memory.pop(0)
            
    def calculate_confidence(self, decision: np.ndarray, stress_level: float) -> float:
        """Calculate confidence in decision"""
        # Lower confidence under high stress
        base_confidence = 1 - stress_level
        
        # Check if similar situations in memory led to good outcomes
        confidence = base_confidence * (1 - np.mean(np.abs(decision - 0.5)))
        
        return float(confidence)

class DualNetwork:
    def __init__(self, input_size: int = 5):
        self.gut = GutNetwork(input_size)
        self.brain = BrainNetwork(input_size)
        
    def process(self, input_signal: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Process input through both networks
        Returns: final decision and metadata
        """
        # First process through gut network
        gut_signal, stress_level = self.gut.process_input(input_signal)
        
        # Then process through brain network
        decision, metadata = self.brain.process_input(input_signal, gut_signal, stress_level)
        
        return decision, metadata

# Example usage and testing
def test_dual_network():
    # Initialize network
    network = DualNetwork(input_size=5)
    
    # Test with normal input
    normal_input = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    decision, metadata = network.process(normal_input)
    print("\nNormal input test:")
    print(f"Decision: {decision}")
    print(f"Metadata: {metadata}")
    
    # Test with extreme input
    extreme_input = np.array([0.9, 0.9, 0.9, 0.9, 0.9])
    decision, metadata = network.process(extreme_input)
    print("\nExtreme input test:")
    print(f"Decision: {decision}")
    print(f"Metadata: {metadata}")
    
    # Test adaptation over time
    print("\nTesting adaptation:")
    for i in range(5):
        decision, metadata = network.process(extreme_input)
        print(f"Iteration {i+1}:")
        print(f"Confidence: {metadata['confidence']:.3f}")
        print(f"Gut influence: {metadata['gut_influence']:.3f}")

if __name__ == "__main__":
    test_dual_network()
