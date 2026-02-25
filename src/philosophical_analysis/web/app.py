"""
Philosophical Text Analysis - Web Interface

A Streamlit-based web interface for analyzing philosophical texts.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys
import json

import logging
logging.basicConfig(level=logging.INFO)

from philosophical_analysis.core.integrated_analyzer import IntegratedPhilosophicalAnalyzer
from philosophical_analysis.visualization.generator import VisualizationGenerator

# Page configuration
st.set_page_config(
    page_title="Philosophical Text Analysis",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Application title and description
st.title("🏛️ Philosophical Text Analysis")
st.markdown("""
    **Analyze philosophical texts using advanced psycholinguistic techniques**
    
    This tool analyzes philosophical texts using Latent Semantic Analysis (LSA) and other techniques
    to discover patterns in different philosophical thinking styles.
""")

# Initialize session state to store analysis results
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = IntegratedPhilosophicalAnalyzer()
    
if 'analyzed_texts' not in st.session_state:
    st.session_state.analyzed_texts = {}
    
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
    
if 'viz_data' not in st.session_state:
    st.session_state.viz_data = None

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page",
    ["Upload & Analyze", "Dashboard", "Coherence Analysis", "Semantic Network", "About"]
)

# Main content based on selected page
if page == "Upload & Analyze":
    st.header("Upload & Analyze Texts")
    
    # Sample texts option
    use_samples = st.checkbox("Use sample texts", value=False)
    
    if use_samples:
        st.info("Using pre-loaded sample philosophical texts")
        # Load sample texts here (this would need to be implemented)
        sample_texts = {
            "kant_sample": "The categorical imperative is the central philosophical concept in Kant's deontological moral philosophy. It is a way of evaluating motivations for action. The categorical imperative is Kant's formulation of the moral law that he believes is binding on all rational beings regardless of empirical considerations.",
            "nietzsche_sample": "God is dead. God remains dead. And we have killed him. How shall we comfort ourselves, the murderers of all murderers? What was holiest and mightiest of all that the world has yet owned has bled to death under our knives: who will wipe this blood off us?",
            "hume_sample": "Reason is, and ought only to be the slave of the passions, and can never pretend to any other office than to serve and obey them. The essence of belief is some sentiment or feeling that does not depend on the will, and which accompanies the idea whenever it is present."
        }
        st.session_state.analyzed_texts = sample_texts
    else:
        # Text upload options
        upload_type = st.radio("Choose input method", ["Upload files", "Paste text"])
        
        if upload_type == "Upload files":
            uploaded_files = st.file_uploader("Upload philosophical text files", 
                                             type=["txt"], 
                                             accept_multiple_files=True)
            
            if uploaded_files:
                texts = {}
                for file in uploaded_files:
                    try:
                        content = file.read().decode("utf-8")
                        texts[file.name.split('.')[0]] = content
                        st.success(f"Successfully loaded: {file.name}")
                    except Exception as e:
                        st.error(f"Failed to load {file.name}: {e}")
                
                if texts:
                    st.session_state.analyzed_texts = texts
        
        else:  # Paste text
            col1, col2 = st.columns(2)
            with col1:
                text_name = st.text_input("Text name (e.g., 'kant_critique')")
            with col2:
                author = st.text_input("Author (optional)")
                
            text_content = st.text_area("Paste philosophical text here", height=300)
            
            if st.button("Add Text") and text_name and text_content:
                if text_name in st.session_state.analyzed_texts:
                    st.warning(f"Text '{text_name}' already exists. Choose a different name.")
                else:
                    st.session_state.analyzed_texts[text_name] = text_content
                    st.success(f"Added text: {text_name}")
    
    # Display loaded texts
    if st.session_state.analyzed_texts:
        st.subheader("Loaded Texts")
        text_df = pd.DataFrame({
            "Text ID": list(st.session_state.analyzed_texts.keys()),
            "Length (chars)": [len(text) for text in st.session_state.analyzed_texts.values()]
        })
        st.dataframe(text_df)
        
        # Analysis options
        st.subheader("Analysis Options")
        lsa_components = st.slider("LSA Components", min_value=5, max_value=100, value=10, 
                                  help="Number of dimensions for LSA analysis")
                                  
        coherence_window = st.slider("Coherence Window", min_value=2, max_value=10, value=5,
                                   help="Window size for temporal coherence analysis")
        
        # Run analysis button
        if st.button("Run Analysis"):
            with st.spinner("Analyzing texts... This may take a moment."):
                try:
                    # Configure analyzer with selected options
                    analyzer = IntegratedPhilosophicalAnalyzer(
                        lsa_components=lsa_components,
                        coherence_window=coherence_window
                    )
                    
                    # Fit the analyzer on all texts
                    analyzer.fit(st.session_state.analyzed_texts)
                    
                    # Run analysis on all texts
                    results = analyzer.analyze_multiple_texts(st.session_state.analyzed_texts)
                    
                    # Store results in session state
                    st.session_state.analyzer = analyzer
                    st.session_state.analysis_results = results
                    
                    # Generate visualization data
                    temp_dir = Path("temp_viz")
                    temp_dir.mkdir(exist_ok=True)
                    
                    viz_gen = VisualizationGenerator(output_dir=str(temp_dir))
                    viz_data = viz_gen.generate_all_visualizations(
                        analysis_results=results,
                        texts=st.session_state.analyzed_texts,
                        save_html=True
                    )
                    
                    st.session_state.viz_data = viz_data
                    
                    st.success("Analysis complete! Navigate to the Dashboard to view results.")
                    
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                    st.exception(e)
    else:
        st.info("Please upload or paste philosophical texts to analyze.")

elif page == "Dashboard":
    st.header("Analysis Dashboard")
    
    if st.session_state.analysis_results is None:
        st.warning("No analysis results available. Please run analysis first.")
    else:
        # Display basic stats
        st.subheader("Analysis Summary")
        results = st.session_state.analysis_results
        
        # Create metrics row
        col1, col2, col3 = st.columns(3)
        with col1:
            avg_coherence = results['first_order_coherence'].mean()
            st.metric("Avg. First-Order Coherence", f"{avg_coherence:.3f}")
        
        with col2:
            if 'second_order_coherence' in results.columns:
                avg_second_order = results['second_order_coherence'].mean()
                st.metric("Avg. Second-Order Coherence", f"{avg_second_order:.3f}")
        
        with col3:
            if 'target_determiners_freq' in results.columns:
                avg_determiners = results['target_determiners_freq'].mean()
                st.metric("Avg. Determiner Frequency", f"{avg_determiners:.3f}")
        
        # Display results table
        st.subheader("Detailed Results")
        display_columns = ['text_id', 'first_order_coherence', 'second_order_coherence', 
                          'target_determiners_freq', 'avg_sentence_length', 
                          'predicted_label', 'classification_confidence']
        
        # Filter columns that actually exist
        valid_columns = [col for col in display_columns if col in results.columns]
        st.dataframe(results[valid_columns])
        
        # Visualization from dashboard data
        if st.session_state.viz_data and 'dashboard' in st.session_state.viz_data:
            dashboard_data = st.session_state.viz_data['dashboard']
            
            st.subheader("Coherence Comparison")
            # Create a bar chart for coherence comparison
            if 'philosophers' in dashboard_data:
                coherence_data = {
                    'Philosopher': [],
                    'First-Order Coherence': []
                }
                
                for philosopher, metrics in dashboard_data['philosophers'].items():
                    coherence_data['Philosopher'].append(philosopher)
                    coherence_data['First-Order Coherence'].append(
                        metrics.get('first_order_coherence', 0)
                    )
                
                coherence_df = pd.DataFrame(coherence_data)
                st.bar_chart(coherence_df.set_index('Philosopher'))

elif page == "Coherence Analysis":
    st.header("Coherence Analysis")
    
    if st.session_state.analysis_results is None:
        st.warning("No analysis results available. Please run analysis first.")
    elif st.session_state.viz_data is None or 'temporal' not in st.session_state.viz_data:
        st.warning("Temporal coherence data not available.")
    else:
        # Display temporal coherence data
        temporal_data = st.session_state.viz_data['temporal']
        
        # Select philosopher to display
        philosophers = list(temporal_data.keys())
        selected_philosopher = st.selectbox("Select philosopher", philosophers)
        
        if selected_philosopher in temporal_data:
            # Display temporal coherence chart
            st.subheader(f"Temporal Coherence for {selected_philosopher}")
            
            philosopher_data = temporal_data[selected_philosopher]
            timeline = philosopher_data.get('coherence_timeline', [])
            
            if timeline:
                df = pd.DataFrame({
                    'Segment': range(len(timeline)),
                    'Coherence': timeline
                })
                st.line_chart(df.set_index('Segment'))
                
                # Display coherence stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Coherence", f"{philosopher_data.get('avg_coherence', 0):.3f}")
                with col2:
                    st.metric("Volatility", f"{philosopher_data.get('volatility', 0):.3f}")
                with col3:
                    st.metric("Trend", f"{philosopher_data.get('trend', 0):.3f}")
            else:
                st.info("No temporal coherence data available for this philosopher.")

elif page == "Semantic Network":
    st.header("Semantic Network")
    
    if st.session_state.analysis_results is None:
        st.warning("No analysis results available. Please run analysis first.")
    elif st.session_state.viz_data is None or 'network' not in st.session_state.viz_data or not st.session_state.viz_data['network']:
        st.warning("Semantic network data not available.")
    else:
        network_data = st.session_state.viz_data['network']
        
        # Display network stats
        if 'metadata' in network_data:
            metadata = network_data['metadata']
            st.subheader("Network Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Concepts", metadata.get('total_concepts', 0))
            with col2:
                st.metric("Total Relationships", metadata.get('total_relationships', 0))
            with col3:
                st.metric("Network Density", f"{metadata.get('density', 0):.3f}")
        
        # Display network visualization placeholder
        # In a real implementation, we'd use a plotly or D3.js graph here
        st.subheader("Semantic Network Visualization")
        st.info("Semantic network visualization will be displayed here.")
        
        # Display top concepts
        if 'nodes' in network_data:
            nodes = network_data['nodes']
            if nodes:
                st.subheader("Top Concepts")
                
                # Sort nodes by importance
                sorted_nodes = sorted(nodes, key=lambda x: x.get('importance', 0), reverse=True)
                top_nodes = sorted_nodes[:10]
                
                node_df = pd.DataFrame([{
                    'Concept': node.get('label', ''),
                    'Philosopher': node.get('philosopher', ''),
                    'Category': node.get('category', ''),
                    'Importance': node.get('importance', 0)
                } for node in top_nodes])
                
                st.dataframe(node_df)

elif page == "About":
    st.header("About Philosophical Text Analysis")
    
    st.markdown("""
    ## 🎯 What This Does

    This project uses **Latent Semantic Analysis (LSA)** and other techniques from the research paper 
    ["Automated analysis of free speech predicts psychosis onset"](https://www.nature.com/articles/npjschz201530) 
    to analyze philosophical texts and discover patterns in different thinking styles.

    ### 🔍 Key Findings 

    **Surprising Results from Initial Analysis:**
    - **Nietzsche** (Beyond Good & Evil): **0.667 coherence** 🥇
    - **Kant** (Critique of Pure Reason): **0.581 coherence** 🥈  
    - **Hume** (Human Understanding): **0.570 coherence** 🥉

    *This challenges assumptions about "systematic" vs "fragmentary" philosophical styles!*
    
    ### 📚 Features
    
    - Advanced textual analysis using LSA and Part-of-Speech processing
    - Coherence measurement across philosophical works
    - Temporal coherence tracking within texts
    - Semantic network visualization of philosophical concepts
    - Comparative analysis across philosophical traditions
    """)
    
    st.markdown("---")
    st.markdown("© 2025 Philosophical Text Analysis Project")


# Run the Streamlit app with:
# streamlit run src/philosophical_analysis/web/app.py
