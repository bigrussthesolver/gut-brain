import streamlit as st
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import json
from llm_dual_network import LLMDualNetwork
import plotly.express as px
import pandas as pd
import os
import uuid
from pathlib import Path
from typing import List, Dict

# Create output directory if it doesn't exist
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Initialize paths
if 'PATHS' not in st.session_state:
    st.session_state.PATHS = {
        'output': Path('output'),
        'analysis': Path('analysis')
    }
    # Create directories if they don't exist
    for path in st.session_state.PATHS.values():
        path.mkdir(exist_ok=True)

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
    
    return filepath

def load_session():
    """Load a session from a file"""
    session_files = list(OUTPUT_DIR.glob("*.json"))
    if not session_files:
        return None
    
    with st.sidebar:
        st.subheader("Load Session")
        session_file = st.selectbox("Select a session file", [file.name for file in session_files])
        if st.button("Load Session"):
            filepath = OUTPUT_DIR / session_file
            with open(filepath, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            st.session_state.chat_history = session_data['chat_history']
            st.session_state.emotional_history = session_data['emotional_history']
            st.session_state.session_id = session_data['session_id']
            st.session_state.session_start_time = session_data['start_time']
            st.success(f"Session loaded from: {filepath.name}")

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

def display_chat_message(role, content, metadata=None):
    """Display a chat message with optional metadata"""
    with st.chat_message(role):
        st.write(content)
        if metadata and role == "assistant":
            with st.expander("View Response Metadata"):
                st.json(metadata)

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

# App title and description
st.title("Emotional AI Interaction System")
st.markdown("""
This system demonstrates an AI that processes both the content and emotional aspects of communication.
The visualization shows the emotional state analysis and how it influences responses.
""")

# Session management buttons in the sidebar
with st.sidebar:
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
            
    # Add analyze button
    if st.button("Analyze Session", key="analyze_session"):
        if not st.session_state.get('chat_history'):
            st.warning("No conversation to analyze yet.")
        else:
            with st.spinner("Analyzing conversation..."):
                # Get analysis
                analysis = st.session_state.network.analyze_session(
                    st.session_state.chat_history,
                    st.session_state.emotional_history
                )
                
                # Save analysis
                analysis_file = save_analysis(analysis)
                st.session_state.session_analysis = analysis
                st.success(f"Analysis saved to: {analysis_file.name}")
                st.rerun()
    
    # Display analysis if available
    if st.session_state.get('session_analysis'):
        st.subheader("Session Analysis")
        st.markdown(st.session_state.session_analysis)
    
    # Display previous analyses
    st.subheader("Previous Analyses")
    analyses = load_session_analyses()
    for analysis in analyses:
        with st.expander(f"Analysis {analysis['timestamp']}"):
            st.markdown(f"**Session ID:** {analysis['session_id']}")
            st.markdown(f"**Messages:** {analysis['chat_history_length']}")
            st.markdown(f"**Session File:** {analysis['session_file']}")
            st.markdown("**Analysis:**")
            st.markdown(analysis['analysis_text'])

    load_session()
    
    # Display current session info
    st.info(f"Session ID: {st.session_state.session_id[:8]}...")
    st.info(f"Started: {datetime.fromisoformat(st.session_state.session_start_time).strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Emotional state visualization
    st.subheader("System State")
    if st.session_state.emotional_history:
        latest_emotional_state = st.session_state.emotional_history[-1]
        radar_chart = create_radar_chart(latest_emotional_state)
        st.plotly_chart(radar_chart, use_container_width=True)
        
        timeline = create_emotion_timeline()
        if timeline:
            st.plotly_chart(timeline, use_container_width=True)
    
    # System stress level
    if st.session_state.chat_history:
        latest_response = st.session_state.chat_history[-1]
        if 'metadata' in latest_response and 'stress_level' in latest_response['metadata']:
            stress_level = latest_response['metadata']['stress_level']
            st.metric("System Stress Level", f"{stress_level:.2f}")

# Handle new session request
if st.session_state.new_session_requested:
    # Clear all state
    for key in list(st.session_state.keys()):
        if key != 'new_session_requested':
            del st.session_state[key]
    
    # Initialize fresh state
    st.session_state.chat_history = []
    st.session_state.emotional_history = []
    st.session_state.network = LLMDualNetwork()
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.session_start_time = datetime.now().isoformat()
    st.session_state.new_session_requested = False
    st.rerun()

# Process new messages
if prompt := st.chat_input("Enter your message..."):
    try:
        # Process through dual network
        response, metadata = st.session_state.network.process(prompt)
        
        # Update session state
        st.session_state.chat_history.extend([
            {
                "role": "user",
                "content": prompt,
                "timestamp": datetime.now().isoformat()
            },
            {
                "role": "assistant",
                "content": response,
                "metadata": metadata,
                "timestamp": datetime.now().isoformat()
            }
        ])
        
        # Store emotional state
        if 'emotional_influence' in metadata:
            st.session_state.emotional_history.append(metadata['emotional_influence'])
            
        # Rerun to update UI with new state
        st.rerun()
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Display chat history
if st.session_state.get('chat_history'):
    for message in st.session_state.chat_history:
        display_chat_message(
            message['role'],
            message['content'],
            message.get('metadata', None)
        )

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
