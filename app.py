import streamlit as st
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import json
from llm_dual_network import LLMDualNetwork, load_env_variables
import plotly.express as px
import pandas as pd
import os
import uuid
from pathlib import Path
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Verify environment variables before starting
if not load_env_variables():
    st.error("Failed to load environment variables. Please check your .env file and ensure OPENAI_API_KEY is set correctly.")
    st.stop()

# Create output directory if it doesn't exist
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
logger.debug(f"Output directory created/verified: {OUTPUT_DIR}")

# Initialize paths
if 'PATHS' not in st.session_state:
    st.session_state.PATHS = {
        'output': Path('output'),
        'analysis': Path('analysis')
    }
    # Create directories if they don't exist
    for path in st.session_state.PATHS.values():
        path.mkdir(exist_ok=True)
    logger.debug("Path directories initialized")

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'emotional_history' not in st.session_state:
    st.session_state.emotional_history = []
if 'network' not in st.session_state:
    st.session_state.network = LLMDualNetwork()
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'session_start_time' not in st.session_state:
    st.session_state.session_start_time = datetime.now().isoformat()

# Check if we need to start a new session
if 'new_session_requested' not in st.session_state:
    st.session_state.new_session_requested = False

def save_session():
    """Save current session to a file"""
    logger.debug(f"Saving session {st.session_state.session_id}")
    session_data = {
        'session_id': st.session_state.session_id,
        'start_time': st.session_state.session_start_time,
        'end_time': datetime.now().isoformat(),
        'chat_history': st.session_state.chat_history,
        'emotional_history': st.session_state.emotional_history
    }
    
    # Create filename with session ID and timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"session_{st.session_state.session_id}_{timestamp}.json"
    filepath = OUTPUT_DIR / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(session_data, f, indent=2, ensure_ascii=False)
    
    logger.debug(f"Session saved to {filepath}")
    return filepath

def load_session(filepath: Path) -> bool:
    """Load a session from a file"""
    try:
        logger.debug(f"Loading session from {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            session_data = json.load(f)
        
        # Save current session if it exists
        if st.session_state.chat_history:
            save_session()
        
        # Load selected session
        st.session_state.chat_history = session_data['chat_history']
        st.session_state.emotional_history = session_data.get('emotional_history', [])
        st.session_state.session_id = session_data['session_id']
        st.session_state.session_start_time = session_data['start_time']
        
        # Create fresh network instance
        st.session_state.network = LLMDualNetwork()
        
        # Restore conversation context in both networks
        logger.debug("Restoring conversation history in networks")
        st.session_state.network.brain.restore_conversation_history(session_data['chat_history'])
        st.session_state.network.gut.restore_conversation_history(session_data['chat_history'])
        
        logger.debug("Session loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error loading session: {str(e)}")
        return False

def reset_session_state():
    """Reset all session state variables to their initial values"""
    logger.debug("Resetting session state")
    try:
        # Verify environment variables before creating new network
        if not load_env_variables():
            raise ValueError("Failed to load environment variables")
            
        # Save current session if it exists
        if st.session_state.get('chat_history'):
            logger.debug("Saving current session before reset")
            save_session()
        
        # Clear all session-specific state
        keys_to_clear = [
            'chat_history', 
            'emotional_history', 
            'network',
            'session_id',
            'session_start_time',
            'session_analysis',
            'new_session_requested'
        ]
        
        logger.debug(f"Clearing session state keys: {keys_to_clear}")
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        
        # Initialize fresh state
        logger.debug("Initializing fresh session state")
        st.session_state.chat_history = []
        st.session_state.emotional_history = []
        st.session_state.network = LLMDualNetwork()
        st.session_state.network.refresh_clients()  # Refresh OpenAI clients with current environment variables
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.session_start_time = datetime.now().isoformat()
        st.session_state.new_session_requested = False
        logger.debug(f"New session initialized with ID: {st.session_state.session_id}")
    except Exception as e:
        logger.error(f"Error resetting session state: {str(e)}")
        st.error(f"Error resetting session state: {str(e)}")
        raise

def validate_message(message: Dict) -> Dict:
    """Validate and fix message format if needed"""
    if not isinstance(message, dict):
        logger.warning(f"Invalid message format: {message}")
        return None
        
    if 'content' not in message:
        logger.warning(f"Message missing content: {message}")
        return None
        
    # Ensure required fields
    if 'role' not in message:
        if 'metadata' in message:
            message['role'] = 'assistant'
        else:
            message['role'] = 'user'
            
    if 'timestamp' not in message:
        message['timestamp'] = datetime.now().isoformat()
        
    return message

def display_chat_message(role: str, content: str, metadata: Dict = None, show_metadata: bool = True):
    """Display a chat message with optional metadata"""
    if role == "user":
        st.chat_message("user").write(content)
    else:
        with st.chat_message("assistant"):
            st.write(content)
            if metadata and show_metadata:
                with st.expander("View Message Details", expanded=False):
                    # Display emotional influence if present
                    if 'emotional_influence' in metadata:
                        emotions = metadata['emotional_influence']
                        cols = st.columns(3)
                        with cols[0]:
                            st.metric("Valence", f"{emotions['valence']:.2f}")
                        with cols[1]:
                            st.metric("Arousal", f"{emotions['arousal']:.2f}")
                        with cols[2]:
                            st.metric("Dominance", f"{emotions['dominance']:.2f}")
                    
                    # Display other metadata
                    st.write("**Full Metadata:**")
                    st.json(metadata)

def display_session_messages(messages: List[Dict], in_expander: bool = False):
    """Display a list of messages, with special handling for expander contexts"""
    for msg in messages:
        if isinstance(msg, dict) and 'content' in msg and 'role' in msg:
            display_chat_message(
                msg['role'],
                msg['content'],
                msg.get('metadata', None),
                show_metadata=not in_expander
            )

def create_radar_chart(emotional_state):
    """Create a radar chart for emotional state visualization"""
    categories = ['Valence', 'Arousal', 'Dominance']
    values = [emotional_state['valence'], 
              emotional_state['arousal'], 
              emotional_state['dominance']]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],  # Complete the polygon
        theta=categories + [categories[0]],  # Complete the polygon
        fill='toself',
        name='Current State'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False,
        title="Emotional State Analysis"
    )
    return fig

def create_emotion_timeline():
    """Create a timeline of emotional states"""
    if not st.session_state.emotional_history:
        return None
        
    df = pd.DataFrame(st.session_state.emotional_history)
    fig = go.Figure()
    
    # Add traces for each emotional dimension
    for dimension in ['valence', 'arousal', 'dominance']:
        fig.add_trace(go.Scatter(
            x=list(range(len(df))),
            y=df[dimension],
            name=dimension.capitalize(),
            mode='lines+markers'
        ))
    
    fig.update_layout(
        title="Emotional State Timeline",
        xaxis_title="Interaction Number",
        yaxis_title="Intensity",
        yaxis=dict(range=[0, 1]),
        height=400
    )
    return fig

def save_analysis(analysis_text: str) -> Path:
    """Save analysis to file with session mapping"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = st.session_state.session_id
    
    # Create analysis data with session mapping
    analysis_data = {
        "session_id": session_id,
        "timestamp": timestamp,
        "analysis_text": analysis_text,
        "session_file": f"session_{session_id}_{timestamp}.json",
        "chat_history_length": len(st.session_state.chat_history),
        "emotional_history_length": len(st.session_state.emotional_history)
    }
    
    # Save analysis to file
    analysis_file = st.session_state.PATHS['analysis'] / f"analysis_{session_id}_{timestamp}.json"
    with open(analysis_file, 'w') as f:
        json.dump(analysis_data, f, indent=2)
    
    return analysis_file

def load_session_analyses() -> List[Dict]:
    """Load all analyses and their session mappings"""
    analyses = []
    analysis_path = st.session_state.PATHS['analysis']
    
    if analysis_path.exists():
        for file in analysis_path.glob("analysis_*.json"):
            with open(file, 'r') as f:
                analysis_data = json.load(f)
                analyses.append(analysis_data)
    
    return sorted(analyses, key=lambda x: x['timestamp'], reverse=True)

# Initialize Streamlit app
st.set_page_config(page_title="Gut-Brain Network", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state if needed
if 'chat_history' not in st.session_state:
    reset_session_state()

# Sidebar UI
with st.sidebar:
    st.title("Settings & Controls")
    
    # Session Management
    st.subheader("Session Management")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("New Session", key="new_session"):
            if st.session_state.chat_history:
                save_session()
            st.session_state.new_session_requested = True
            st.rerun()
    with col2:
        if st.button("Save Session", key="save_session"):
            filepath = save_session()
            st.success(f"Session saved to: {filepath.name}")
    
    # Load Previous Session
    st.subheader("Load Session")
    session_files = list(OUTPUT_DIR.glob("session_*.json"))
    if session_files:
        selected_file = st.selectbox(
            "Select session to load",
            options=session_files,
            format_func=lambda x: f"Session {x.stem.split('_')[1]} ({x.stem.split('_')[2]})",
            key="load_session_select"
        )
        
        if st.button("Load Selected Session", key="load_session"):
            if load_session(selected_file):
                st.success(f"Loaded session from {selected_file.name}")
                st.rerun()
            else:
                st.error("Failed to load session. Check the logs for details.")
    else:
        st.info("No saved sessions found")
    
    # Display current session info
    st.subheader("Current Session")
    st.info(f"Session ID: {st.session_state.session_id[:8]}...")
    st.info(f"Started: {datetime.fromisoformat(st.session_state.session_start_time).strftime('%Y-%m-%d %H:%M:%S')}")
    
    # System stress level
    if st.session_state.chat_history:
        latest_response = st.session_state.chat_history[-1]
        if isinstance(latest_response, dict) and 'metadata' in latest_response:
            metadata = latest_response['metadata']
            if isinstance(metadata, dict) and 'stress_level' in metadata:
                stress_level = metadata['stress_level']
                st.metric("System Stress Level", f"{stress_level:.2f}")
    
    # Quick Analysis
    st.subheader("Quick Analysis")
    if st.button("Generate Analysis", key="quick_analysis"):
        if not st.session_state.get('chat_history'):
            st.warning("No conversation to analyze yet.")
        else:
            with st.spinner("Analyzing conversation..."):
                analysis = st.session_state.network.analyze_session(
                    st.session_state.chat_history,
                    st.session_state.emotional_history
                )
                # Save analysis
                analysis_file = save_analysis(analysis)
                st.session_state.session_analysis = analysis
                st.success(f"Analysis saved to: {analysis_file.name}")
                st.rerun()

# Create tabs
tab_chat, tab_history = st.tabs(["Chat", "History & Analysis"])

with tab_chat:
    # Header with emotional state visualization
    st.title("Gut-Brain Network")
    
    st.markdown("""
    This AI integrates both emotional (gut) and rational (brain) processing to provide more nuanced and context-aware responses.
    """)
    
    # Display current emotional state if available
    if st.session_state.get('emotional_history'):
        current_emotional_state = st.session_state.emotional_history[-1]
        fig = create_radar_chart(current_emotional_state)
        st.plotly_chart(fig, use_container_width=True)
    
    # Process new messages
    if prompt := st.chat_input("Enter your message..."):
        try:
            logger.debug(f"Processing new message: {prompt[:50]}...")
            # Process through dual network
            response, metadata = st.session_state.network.process(prompt)
            
            # Update session state
            timestamp = datetime.now().isoformat()
            
            # Add user message
            user_message = {
                "role": "user",
                "content": prompt,
                "timestamp": timestamp
            }
            
            # Add assistant message
            assistant_message = {
                "role": "assistant",
                "content": response,
                "metadata": metadata,
                "timestamp": timestamp
            }
            
            # Update chat history
            st.session_state.chat_history.extend([user_message, assistant_message])
            
            # Store emotional state
            if 'emotional_influence' in metadata:
                st.session_state.emotional_history.append(metadata['emotional_influence'])
                
            # Rerun to update UI with new state
            st.rerun()
            
        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")
            st.error(f"An error occurred: {str(e)}")
    
    # Display chat history
    if st.session_state.get('chat_history'):
        display_session_messages(st.session_state.chat_history)

with tab_history:
    st.title("Session History & Analysis")
    
    # Session Information
    st.subheader("Current Session Info")
    st.write(f"Session ID: {st.session_state.session_id}")
    st.write(f"Started: {st.session_state.session_start_time}")
    
    # Emotional Trajectory
    if st.session_state.get('emotional_history'):
        st.subheader("Emotional Trajectory")
        
        # Create emotional trajectory plot
        fig = go.Figure()
        
        # Extract timestamps and values
        timestamps = [msg['timestamp'] for msg in st.session_state.chat_history if msg['role'] == 'assistant'][-len(st.session_state.emotional_history):]
        
        # Add traces for each emotional dimension
        for dimension in ['valence', 'arousal', 'dominance']:
            values = [e[dimension] for e in st.session_state.emotional_history]
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=values,
                name=dimension.capitalize(),
                mode='lines+markers'
            ))
        
        fig.update_layout(
            title="Emotional Dimensions Over Time",
            xaxis_title="Time",
            yaxis_title="Value",
            yaxis_range=[0, 1],
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Session Analysis
    st.subheader("Detailed Analysis")
    if len(st.session_state.chat_history) > 1:
        if st.button("Generate Detailed Analysis", key="detailed_analysis"):
            with st.spinner("Analyzing conversation..."):
                analysis = st.session_state.network.analyze_session(
                    st.session_state.chat_history,
                    st.session_state.emotional_history
                )
                st.markdown(analysis)
    else:
        st.info("Start a conversation to generate analysis.")
    
    # Previous Sessions
    st.subheader("Previous Sessions")
    session_files = list(OUTPUT_DIR.glob("session_*.json"))
    if session_files:
        selected_session = st.selectbox(
            "Select a session to view",
            options=session_files,
            format_func=lambda x: f"Session {x.stem.split('_')[1]} ({x.stem.split('_')[2]})",
            key="view_session_select"
        )
        
        if selected_session:
            with open(selected_session, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
                
            st.write(f"Session ID: {session_data['session_id']}")
            st.write(f"Start Time: {session_data['start_time']}")
            st.write(f"End Time: {session_data['end_time']}")
            
            with st.expander("View Conversation"):
                display_session_messages(session_data['chat_history'], in_expander=True)

# Footer with system information
st.markdown("---")
with st.expander("About this System"):
    st.markdown("""
    This interface demonstrates the LLM-Enhanced Dual Network System, which combines:
    - Emotional state analysis using GPT-4
    - Adaptive response generation
    - Real-time visualization of system state
    - Conversation history tracking
    
    The radar chart shows the current emotional state analysis, while the timeline
    displays the evolution of emotional states throughout the conversation.
    
    Sessions are automatically saved in the 'output' directory with timestamps
    for future reference and analysis.
    """)

# Handle new session request
if st.session_state.new_session_requested:
    reset_session_state()
    st.rerun()
