import numpy as np
from typing import List, Tuple, Dict, Optional
import json
from datetime import datetime
import time
import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_env_variables():
    """Load environment variables from .env file"""
    dotenv_path = find_dotenv()
    if not dotenv_path:
        logger.error("No .env file found")
        return False
    
    logger.debug(f"Loading environment variables from {dotenv_path}")
    load_dotenv(dotenv_path, override=True)
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment variables")
        return False
    
    logger.debug(f"Environment variables loaded successfully. API key: {api_key[:8]}...")
    return True

def get_openai_client():
    """Get a fresh OpenAI client with the current API key"""
    if not load_env_variables():
        raise ValueError("Failed to load environment variables")
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found after reloading environment")
        
    logger.debug(f"Creating new OpenAI client with API key: {api_key[:8]}...")
    return OpenAI(api_key=api_key)

# Initial environment load
load_env_variables()

class LLMGutNetwork:
    def __init__(self, state_size: int = 5):
        logger.debug("Initializing LLMGutNetwork")
        self.state_size = state_size
        self.internal_state = np.zeros(state_size)
        self.homeostasis_target = np.ones(state_size) * 0.5
        self.adaptation_rate = 0.1
        self.stress_threshold = 0.7
        self.emotional_memory = []
        self.max_memory_size = 50
        self.client = get_openai_client()
        logger.debug("LLMGutNetwork initialized")
    
    def refresh_client(self):
        """Refresh the OpenAI client with current environment variables"""
        logger.debug("Refreshing OpenAI client for LLMGutNetwork")
        self.client = get_openai_client()
        
    def process_emotional_content(self, text: str) -> Dict[str, float]:
        """Analyze emotional content of text using GPT-4"""
        logger.debug(f"Processing emotional content: {text[:50]}...")
        try:
            response = self.client.chat.completions.create(
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
            logger.error(f"Error in emotional analysis: {e}")
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
        logger.debug(f"Processing input: {text_input[:50]}...")
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
    
    def restore_emotional_memory(self, emotional_history: List[Dict]):
        """Restore emotional memory from a saved session"""
        logger.debug(f"Restoring emotional memory with {len(emotional_history)} states")
        self.emotional_memory = emotional_history
        
        # Update internal state based on the most recent emotional state
        if emotional_history:
            latest_state = emotional_history[-1]
            self.internal_state = np.array([
                latest_state['valence'],
                latest_state['arousal'],
                latest_state['dominance'],
                0.5,  # Default values for other dimensions
                0.5
            ])
            logger.debug(f"Restored internal state: {self.internal_state}")

class LLMBrainNetwork:
    def __init__(self, system_prompt: str = "You are a helpful assistant."):
        logger.debug("Initializing LLMBrainNetwork")
        self.system_prompt = system_prompt
        self.conversation_history = []
        self.max_history = 10
        self.confidence_threshold = 0.6
        self.reflection_log = []
        self.client = get_openai_client()
        logger.debug("LLMBrainNetwork initialized")
    
    def refresh_client(self):
        """Refresh the OpenAI client with current environment variables"""
        logger.debug("Refreshing OpenAI client for LLMBrainNetwork")
        self.client = get_openai_client()
        
    def restore_conversation_history(self, history: List[Dict]):
        """Restore conversation history from a saved session"""
        logger.debug(f"Restoring conversation history with {len(history)} messages")
        # Validate and convert history format if needed
        validated_history = []
        for msg in history:
            if isinstance(msg, dict) and 'content' in msg:
                # If it's already in the correct format, just validate required fields
                if 'role' not in msg:
                    # Try to infer role from structure
                    if 'metadata' in msg:
                        msg['role'] = 'assistant'
                    else:
                        msg['role'] = 'user'
                if 'timestamp' not in msg:
                    msg['timestamp'] = datetime.now().isoformat()
                validated_history.append(msg)
            else:
                logger.warning(f"Skipping invalid message format: {msg}")
        
        self.conversation_history = validated_history
        logger.debug(f"Restored {len(self.conversation_history)} conversation entries")
    
    def generate_response(self, 
                         user_input: str, 
                         emotional_state: Dict[str, float], 
                         stress_level: float) -> Tuple[str, Dict]:
        """Generate response using GPT-4, considering emotional state"""
        logger.debug(f"Generating response for input: {user_input[:50]}...")
        try:
            # Prepare conversation context
            emotional_context = (
                f"Current emotional state: valence={emotional_state['valence']:.2f}, "
                f"arousal={emotional_state['arousal']:.2f}, "
                f"dominance={emotional_state['dominance']:.2f}. "
                f"Stress level: {stress_level:.2f}"
            )
            
            messages = [
                {"role": "system", "content": f"{self.system_prompt}\n\nContext: {emotional_context}"}
            ]
            
            # Add conversation history
            for entry in self.conversation_history[-self.max_history:]:
                messages.append({"role": entry['role'], "content": entry['content']})
            
            # Add current user input
            messages.append({"role": "user", "content": user_input})
            
            response = self.client.chat.completions.create(
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
            logger.error(f"Error in response generation: {e}")
            return f"I apologize, but I'm having trouble processing that right now.", {
                'confidence': 0.1,
                'emotional_influence': emotional_state,
                'stress_level': stress_level,
                'error': str(e)
            }
    
    def update_history(self, user_input: str, response: str, metadata: Dict):
        """Maintain conversation history with metadata"""
        logger.debug(f"Updating conversation history with input: {user_input[:50]}...")
        timestamp = datetime.now().isoformat()
        
        # Add user message
        user_message = {
            "role": "user",
            "content": user_input,
            "timestamp": timestamp
        }
        
        # Add assistant message
        assistant_message = {
            "role": "assistant",
            "content": response,
            "metadata": metadata,
            "timestamp": timestamp
        }
        
        self.conversation_history.extend([user_message, assistant_message])
        
        # Trim history if needed
        if len(self.conversation_history) > self.max_history * 2:  # *2 because each exchange has 2 messages
            self.conversation_history = self.conversation_history[-self.max_history * 2:]
    
    def reflect_on_conversation(self) -> Dict:
        """Analyze conversation patterns and generate insights"""
        logger.debug("Reflecting on conversation...")
        if not self.conversation_history:
            return {'insights': 'No conversation history available'}
        
        # Analyze emotional trajectory
        emotional_trajectory = [
            entry['metadata']['emotional_influence']['valence'] 
            for entry in self.conversation_history
            if entry['role'] == 'assistant' and 'metadata' in entry
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
        logger.debug(f"Calculating confidence for response: {response[:50]}...")
        # Lower confidence under high stress
        base_confidence = 1 - stress_level
        
        # Adjust based on response characteristics
        response_length_factor = min(len(response) / 100, 1.0)
        
        confidence = base_confidence * response_length_factor
        return float(confidence)
    
    def restore_conversation_history(self, history: List[Dict]):
        """Restore conversation history from a saved session"""
        logger.debug(f"Restoring conversation history with {len(history)} messages")
        # Validate and convert history format if needed
        validated_history = []
        for msg in history:
            if isinstance(msg, dict) and 'content' in msg:
                # If it's already in the correct format, just validate required fields
                if 'role' not in msg:
                    # Try to infer role from structure
                    if 'metadata' in msg:
                        msg['role'] = 'assistant'
                    else:
                        msg['role'] = 'user'
                if 'timestamp' not in msg:
                    msg['timestamp'] = datetime.now().isoformat()
                validated_history.append(msg)
            else:
                logger.warning(f"Skipping invalid message format: {msg}")
        
        self.conversation_history = validated_history
        logger.debug(f"Restored {len(self.conversation_history)} conversation entries")

class LLMDualNetwork:
    def __init__(self, system_prompt: str = "You are a helpful assistant."):
        logger.debug("Initializing LLMDualNetwork")
        self.gut = LLMGutNetwork()
        self.brain = LLMBrainNetwork(system_prompt)
        self.refresh_clients()
        logger.debug("LLMDualNetwork initialized")
    
    def refresh_clients(self):
        """Refresh all OpenAI clients with current environment variables"""
        logger.debug("Refreshing all OpenAI clients")
        self.gut.refresh_client()
        self.brain.refresh_client()
    
    def process(self, user_input: str) -> Tuple[str, Dict]:
        """Process input through both networks and generate response"""
        logger.debug(f"Processing user input: {user_input[:50]}...")
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
        logger.debug("Starting session analysis")
        if not chat_history:
            return "No conversation to analyze yet."
        
        try:
            # Calculate emotional statistics
            if emotional_history:
                logger.debug(f"Analyzing emotional history with {len(emotional_history)} entries")
                # Calculate averages for emotional dimensions
                avg_valence = sum(e.get('valence', 0.5) for e in emotional_history) / len(emotional_history)
                avg_arousal = sum(e.get('arousal', 0.5) for e in emotional_history) / len(emotional_history)
                avg_dominance = sum(e.get('dominance', 0.5) for e in emotional_history) / len(emotional_history)
                
                # Calculate emotional ranges
                valence_range = max(e.get('valence', 0.5) for e in emotional_history) - min(e.get('valence', 0.5) for e in emotional_history)
                arousal_range = max(e.get('arousal', 0.5) for e in emotional_history) - min(e.get('arousal', 0.5) for e in emotional_history)
                dominance_range = max(e.get('dominance', 0.5) for e in emotional_history) - min(e.get('dominance', 0.5) for e in emotional_history)
            else:
                logger.debug("No emotional history available")
                avg_valence = avg_arousal = avg_dominance = 0.5
                valence_range = arousal_range = dominance_range = 0.0
            
            # Prepare conversation summary
            messages = []
            for msg in chat_history:
                if isinstance(msg, dict) and 'content' in msg and 'role' in msg:
                    messages.append(f"{msg['role']}: {msg['content']}")
            
            conversation_text = "\n".join(messages[-10:])  # Last 10 messages
            
            # Create analysis prompt
            analysis_prompt = f"""
            Analyze this conversation and emotional patterns:
            
            Conversation (last 10 messages):
            {conversation_text}
            
            Emotional Statistics:
            - Average Valence (positivity): {avg_valence:.2f}
            - Average Arousal (energy): {avg_arousal:.2f}
            - Average Dominance (control): {avg_dominance:.2f}
            - Emotional Ranges:
              * Valence Range: {valence_range:.2f}
              * Arousal Range: {arousal_range:.2f}
              * Dominance Range: {dominance_range:.2f}
            
            Please provide:
            1. Overall conversation tone and quality
            2. Emotional patterns and significant shifts
            3. Suggestions for improving interaction quality
            """
            
            client = get_openai_client()
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert in conversation analysis and emotional intelligence."},
                    {"role": "user", "content": analysis_prompt}
                ]
            )
            
            logger.debug("Analysis completed successfully")
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            return f"Analysis failed: {str(e)}"
    
    def restore_session_state(self, chat_history: List[Dict], emotional_history: List[Dict]):
        """Restore the network state from a saved session"""
        logger.debug("Restoring session state")
        self.brain.restore_conversation_history(chat_history)
        self.gut.restore_emotional_memory(emotional_history)
        logger.debug("Session state restored")

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
        logger.debug(f"Processing test input: {input_text[:50]}...")
        print(f"\nProcessing: {input_text}")
        response, metadata = network.process(input_text)
        print(f"Response: {response}")
        print(f"Metadata: {json.dumps(metadata, indent=2)}")
        
        # Add a small delay to simulate real-world timing
        time.sleep(1)
    
    # Generate reflection insights
    reflection = network.brain.reflect_on_conversation()
    logger.debug("Generating conversation reflection...")
    print("\nConversation Reflection:")
    print(json.dumps(reflection, indent=2))

if __name__ == "__main__":
    test_llm_dual_network()
