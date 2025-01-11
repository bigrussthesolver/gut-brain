import numpy as np
from typing import List, Tuple, Dict, Optional
import json
from datetime import datetime
import time
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

class LLMGutNetwork:
    def __init__(self, state_size: int = 5):
        self.state_size = state_size
        self.internal_state = np.zeros(state_size)
        self.homeostasis_target = np.ones(state_size) * 0.5
        self.adaptation_rate = 0.1
        self.stress_threshold = 0.7
        self.emotional_memory = []
        self.max_memory_size = 50
        
    def process_emotional_content(self, text: str) -> Dict[str, float]:
        """Analyze emotional content of text using GPT-4"""
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You will analyze the emotional content of text and output emotional scores."},
                    {"role": "user", "content": text}
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "emotional_analysis",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "valence": {
                                    "type": "number",
                                    "description": "Positive/negative emotional score (0-1)"
                                },
                                "arousal": {
                                    "type": "number",
                                    "description": "High/low energy level (0-1)"
                                },
                                "dominance": {
                                    "type": "number",
                                    "description": "Feeling of control/power (0-1)"
                                }
                            },
                            "required": ["valence", "arousal", "dominance"],
                            "additionalProperties": False
                        },
                        "strict": True
                    }
                }
            )
            
            # Parse the JSON response
            emotional_state = json.loads(response.choices[0].message.content)
            
            # Ensure values are within bounds
            for key in emotional_state:
                emotional_state[key] = max(0, min(1, float(emotional_state[key])))
                
            return emotional_state
            
        except Exception as e:
            print(f"Error in emotional analysis: {e}")
            return {'valence': 0.5, 'arousal': 0.5, 'dominance': 0.5}
    
    def update_emotional_memory(self, text: str, emotional_state: Dict[str, float]):
        """Store emotional experiences"""
        memory_entry = {
            'timestamp': datetime.now().isoformat(),
            'text': text,
            'emotional_state': emotional_state
        }
        self.emotional_memory.append(memory_entry)
        
        if len(self.emotional_memory) > self.max_memory_size:
            self.emotional_memory.pop(0)
    
    def process_input(self, text_input: str) -> Tuple[Dict[str, float], float]:
        """Process text input and update internal state"""
        emotional_state = self.process_emotional_content(text_input)
        
        # Convert emotional state to internal state vector
        new_state = np.array([
            emotional_state['valence'],
            emotional_state['arousal'],
            emotional_state['dominance'],
            np.mean(list(emotional_state.values())),
            time.time() % 1  # Add time-based variation
        ])
        
        # Update internal state
        delta = new_state - self.internal_state
        self.internal_state += self.adaptation_rate * delta
        
        # Calculate stress level
        stress_level = np.mean(np.abs(self.internal_state - self.homeostasis_target))
        
        # Update emotional memory
        self.update_emotional_memory(text_input, emotional_state)
        
        return emotional_state, stress_level

class LLMBrainNetwork:
    def __init__(self, system_prompt: str = "You are a helpful assistant."):
        self.system_prompt = system_prompt
        self.conversation_history = []
        self.max_history = 10
        self.confidence_threshold = 0.6
        self.reflection_log = []
    
    def generate_response(self, 
                         user_input: str, 
                         emotional_state: Dict[str, float], 
                         stress_level: float) -> Tuple[str, Dict]:
        """Generate response using GPT-4, considering emotional state"""
        try:
            # Prepare conversation context
            emotional_context = (
                f"Current emotional state: valence={emotional_state['valence']:.2f}, "
                f"arousal={emotional_state['arousal']:.2f}, "
                f"dominance={emotional_state['dominance']:.2f}. "
                f"Stress level: {stress_level:.2f}"
            )
            
            messages = [
                {"role": "system", "content": f"{self.system_prompt}\n\nContext: {emotional_context}"},
                {"role": "user", "content": user_input}
            ]
            
            # Add relevant conversation history
            for entry in self.conversation_history[-2:]:  # Last 2 exchanges
                messages.append({"role": "user", "content": entry['user_input']})
                messages.append({"role": "assistant", "content": entry['response']})
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response_format",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "response_text": {
                                    "type": "string",
                                    "description": "The response text to the user"
                                },
                                "emotional_awareness": {
                                    "type": "object",
                                    "properties": {
                                        "detected_emotion": {
                                            "type": "string",
                                            "description": "Primary emotion detected in user's message"
                                        },
                                        "empathy_level": {
                                            "type": "number",
                                            "description": "Level of empathy to express (0-1)"
                                        }
                                    },
                                    "required": ["detected_emotion", "empathy_level"],
                                    "additionalProperties": False
                                }
                            },
                            "required": ["response_text", "emotional_awareness"],
                            "additionalProperties": False
                        },
                        "strict": True
                    }
                }
            )
            
            result = json.loads(response.choices[0].message.content)
            
            metadata = {
                'confidence': self.calculate_confidence(result['response_text'], stress_level),
                'emotional_influence': emotional_state,
                'stress_level': stress_level,
                'model_used': 'gpt-4o',
                'emotional_awareness': result['emotional_awareness']
            }
            
            # Update conversation history
            self.update_history(user_input, result['response_text'], metadata)
            
            return result['response_text'], metadata
            
        except Exception as e:
            print(f"Error in response generation: {e}")
            return f"I apologize, but I'm having trouble processing that right now.", {
                'confidence': 0.1,
                'emotional_influence': emotional_state,
                'stress_level': stress_level,
                'error': str(e)
            }
    
    def update_history(self, user_input: str, response: str, metadata: Dict):
        """Maintain conversation history with metadata"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'response': response,
            'metadata': metadata
        }
        
        self.conversation_history.append(entry)
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)
    
    def reflect_on_conversation(self) -> Dict:
        """Analyze conversation patterns and generate insights"""
        if not self.conversation_history:
            return {'insights': 'No conversation history available'}
        
        # Analyze emotional trajectory
        emotional_trajectory = [
            entry['metadata']['emotional_influence']['valence'] 
            for entry in self.conversation_history
        ]
        
        avg_emotion = np.mean(emotional_trajectory)
        emotion_stability = np.std(emotional_trajectory)
        
        reflection = {
            'average_emotional_state': float(avg_emotion),
            'emotional_stability': float(emotion_stability),
            'conversation_length': len(self.conversation_history),
            'timestamp': datetime.now().isoformat()
        }
        
        self.reflection_log.append(reflection)
        return reflection
    
    def calculate_confidence(self, response: str, stress_level: float) -> float:
        """Calculate confidence in the generated response"""
        # Lower confidence under high stress
        base_confidence = 1 - stress_level
        
        # Adjust based on response characteristics
        response_length_factor = min(len(response) / 100, 1.0)
        
        confidence = base_confidence * response_length_factor
        return float(confidence)

class LLMDualNetwork:
    def __init__(self, system_prompt: str = "You are a helpful assistant."):
        self.gut = LLMGutNetwork()
        self.brain = LLMBrainNetwork(system_prompt)
        self.llm = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    def process(self, user_input: str) -> Tuple[str, Dict]:
        """
        Process input through both networks and generate response
        """
        # Process through gut network first
        emotional_state, stress_level = self.gut.process_input(user_input)
        
        # Then process through brain network
        response, metadata = self.brain.generate_response(
            user_input, emotional_state, stress_level
        )
        
        # Add gut network's state to metadata
        metadata['gut_state'] = {
            'internal_state': self.gut.internal_state.tolist(),
            'stress_level': stress_level
        }
        
        return response, metadata
    
    def analyze_session(self, chat_history, emotional_history):
        """Analyze the session using GPT-4 to provide insights about the conversation and emotional patterns"""
        if not chat_history:
            return "No conversation to analyze yet."
        
        # Prepare conversation summary
        conversation = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}" 
            for msg in chat_history
        ])
        
        # Prepare emotional data summary
        emotional_patterns = "No emotional data available."
        if emotional_history:
            avg_valence = sum(e['valence'] for e in emotional_history) / len(emotional_history)
            avg_arousal = sum(e['arousal'] for e in emotional_history) / len(emotional_history)
            avg_dominance = sum(e['dominance'] for e in emotional_history) / len(emotional_history)
            emotional_patterns = f"""
            Average emotional measurements:
            - Valence (positive/negative): {avg_valence:.2f}
            - Arousal (active/passive): {avg_arousal:.2f}
            - Dominance (dominant/submissive): {avg_dominance:.2f}
            """
        
        prompt = f"""
        Analyze this conversation and emotional patterns. Provide insights about:
        1. Main themes and topics discussed
        2. Overall emotional tone and progression
        3. Key interaction patterns
        4. Notable moments or shifts in the conversation
        
        CONVERSATION:
        {conversation}
        
        EMOTIONAL PATTERNS:
        {emotional_patterns}
        
        Provide a concise but insightful analysis that helps understand the interaction dynamics.
        """
        
        try:
            response = self.llm.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert in conversation analysis and emotional intelligence."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Analysis failed: {str(e)}"

def test_llm_dual_network():
    # Initialize network
    network = LLMDualNetwork()
    
    # Test with different types of input
    test_inputs = [
        "I'm really excited about this new project!",
        "I'm worried about the upcoming deadline...",
        "The results are quite disappointing.",
        "Everything is working perfectly now!"
    ]
    
    for input_text in test_inputs:
        print(f"\nProcessing: {input_text}")
        response, metadata = network.process(input_text)
        print(f"Response: {response}")
        print(f"Metadata: {json.dumps(metadata, indent=2)}")
        
        # Add a small delay to simulate real-world timing
        time.sleep(1)
    
    # Generate reflection insights
    reflection = network.brain.reflect_on_conversation()
    print("\nConversation Reflection:")
    print(json.dumps(reflection, indent=2))

if __name__ == "__main__":
    test_llm_dual_network()
