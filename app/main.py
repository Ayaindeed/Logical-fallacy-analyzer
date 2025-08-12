#Libraries
import streamlit as st
import os
import pandas as pd
import warnings
from datetime import datetime
import json
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Optional, Tuple
import re
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.utilities import GoogleSerperAPIWrapper

# Suppress warnings
warnings.filterwarnings('ignore')


class FallacyAnalyzer:
    
    def __init__(self, openai_api_key: str, serper_api_key: str, model_name: str = "gpt-4o-mini"):
        self.openai_api_key = openai_api_key
        self.serper_api_key = serper_api_key
        self.model_name = model_name
        self.fallacies_df = None
        self.analysis_history = []
        
    def load_fallacies(self) -> Optional[pd.DataFrame]:
        """Load fallacies data"""
        try:
            self.fallacies_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'fallacies.csv'))
            return self.fallacies_df
        except FileNotFoundError:
            st.error("fallacies.csv not found!!")
            return None
    
    def initialize_model(self, temperature: float = 0.0, max_tokens: int = 1600) -> ChatOpenAI:
        """Initialize the ChatOpenAI model"""
        return ChatOpenAI(
            temperature=temperature,
            model=self.model_name,
            max_tokens=max_tokens,
            openai_api_key=self.openai_api_key
        )
    
    def create_templates(self) -> Tuple[str, str, str]:
        
        # Template 1:  summarization
        template1 = """You are a communication expert. Given this news article, provide:
1. A clear 5-sentence summary
2. Key claims made in the article
3. Evidence presented (if any)
4. Emotional language used

Be accurate and objective. Do not make things up.

Fallacies to check:
{fallacies_df}

Article: {content}

Analysis:"""

        # Template 2:  fallacy detection
        template2 = """You are an ethics professor analyzing this article summary. Provide a detailed analysis:

1. **Primary Fallacy**: The most significant logical fallacy found
2. **Secondary Fallacies**: Other fallacies present (if any)
3. **Confidence Score**: Rate your confidence (1-10) in the primary fallacy identification
4. **Impact Assessment**: How this fallacy might mislead readers
5. **Alternative Perspective**: A counterpoint to the fallacy identification
6. **Recommendations**: How the article could be improved

Article summary: {summary}
Fallacies to consider: 
{fallacies_df}

Detailed Analysis:"""

        # Template 3: Credibility assessment
        template3 = """You are a media literacy expert. Assess the credibility of this article based on:

1. **Source Quality**: Assessment of the publication
2. **Evidence Quality**: Strength of evidence presented
3. **Bias Indicators**: Signs of potential bias
4. **Missing Context**: What important information might be missing
5. **Overall Credibility Score**: Rate 1-10 with justification

Article summary: {summary}
Original URL: {url}

Credibility Assessment:"""
        
        return template1, template2, template3
    
    def analyze_multiple_articles(self, search_topic: str, num_articles: int = 3) -> List[Dict]:
        """Analyze multiple articles for comparison"""
        
        search = GoogleSerperAPIWrapper(
            type="news",
            tbs="qdr:m1",
            serper_api_key=self.serper_api_key
        )
        
        search_results = search.results(search_topic)
        articles = search_results.get('news', [])[:num_articles]
        
        results = []
        for i, article in enumerate(articles):
            try:
                result = self.analyze_single_article(article['link'], article['title'])
                result['rank'] = i + 1
                results.append(result)
            except Exception as e:
                st.warning(f"Failed to analyze article {i+1}: {str(e)}")
                continue
        
        return results
    
    def analyze_single_article(self, url: str, title: str) -> Dict:
        """Analyze a single article"""
        
        # Load article
        loader = WebBaseLoader(url)
        article_text = ' '.join(loader.load()[0].page_content[:3000].split())
        
        # Initialize model
        llm = self.initialize_model()
        
        # Create templates
        template1, template2, template3 = self.create_templates()
        
        # Create chains
        chain1 = LLMChain(
            llm=llm,
            prompt=PromptTemplate(
                input_variables=["content", "fallacies_df"],
                template=template1
            )
        )
        
        chain2 = LLMChain(
            llm=llm,
            prompt=PromptTemplate(
                input_variables=["summary", "fallacies_df"],
                template=template2
            )
        )
        
        chain3 = LLMChain(
            llm=llm,
            prompt=PromptTemplate(
                input_variables=["summary", "url"],
                template=template3
            )
        )
        
        # Run analysis
        summary = chain1.run(content=article_text, fallacies_df=self.fallacies_df)
        fallacy_analysis = chain2.run(summary=summary, fallacies_df=self.fallacies_df)
        credibility_analysis = chain3.run(summary=summary, url=url)
        
        # Extract confidence score using regex
        confidence_match = re.search(r'confidence.*?(\d+)', fallacy_analysis.lower())
        confidence_score = int(confidence_match.group(1)) if confidence_match else 5
        
        result = {
            'title': title,
            'url': url,
            'summary': summary,
            'fallacy_analysis': fallacy_analysis,
            'credibility_analysis': credibility_analysis,
            'confidence_score': confidence_score,
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def create_visualization(self, results: List[Dict]) -> go.Figure:
        """Create visualization of analysis results"""
        
        if not results:
            return None
        
        # Extract data for visualization
        titles = [r['title'][:50] + "..." for r in results]
        confidence_scores = [r['confidence_score'] for r in results]
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=titles,
                y=confidence_scores,
                marker_color=['red' if score >= 7 else 'orange' if score >= 5 else 'green' 
                             for score in confidence_scores]
            )
        ])
        
        fig.update_layout(
            title="Fallacy Detection Confidence Scores",
            xaxis_title="Articles",
            yaxis_title="Confidence Score (1-10)",
            showlegend=False
        )
        
        return fig

# Streamlit App
def main():
    
    st.set_page_config(
        page_title="Advanced Fallacy Analyzer",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Force dark mode
    st.markdown("""
    <style>
        /* Force dark mode globally */
        .stApp {
            background-color: #0e1117;
            color: #fafafa;
        }
        
        /* Main background - Professional dark gradient */
        .main > div {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
        }
        
        /* Content container - Dark theme */
        .block-container {
            background: rgba(20, 25, 40, 0.95);
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
            padding: 2rem;
            backdrop-filter: blur(10px);
            margin: 2rem auto;
            max-width: 1200px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        /* Header styling - Dark theme */
        .main-title {
            font-size: 2.8rem;
            font-weight: 700;
            color: #ffffff;
            text-align: center;
            margin-bottom: 0.5rem;
            background: linear-gradient(45deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        }
        
        .subtitle {
            font-size: 1.2rem;
            color: #b0b0b0;
            text-align: center;
            margin-bottom: 2rem;
            font-weight: 300;
        }
        
        /* Professional header section - Dark theme */
        .header-section {
            background: rgba(30, 35, 50, 0.8);
            border-radius: 10px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
        }
        
        /* Sidebar styling - Dark theme */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        }
        
        section[data-testid="stSidebar"] > div {
            background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        }
        
        /* Card styling - Dark theme */
        .analysis-card, .info-card {
            background: rgba(30, 35, 50, 0.9);
            border-radius: 10px;
            padding: 1.5rem;
            margin: 1rem 0;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(10px);
        }
        
        .info-card {
            border-left: 4px solid #667eea;
        }
        
        /* Button styling - Dark theme */
        .stButton > button {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.75rem 2rem;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }
        
        /* Input styling - Dark theme */
        .stTextInput > div > div > input,
        .stSelectbox > div > div > select {
            background: rgba(30, 35, 50, 0.9);
            border: 2px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            color: #ffffff;
            transition: all 0.3s ease;
            backdrop-filter: blur(10px);
        }
        
        .stTextInput > div > div > input:focus,
        .stSelectbox > div > div > select:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.3);
        }
        
        /* Metrics styling - Dark theme */
        [data-testid="metric-container"] {
            background: rgba(30, 35, 50, 0.9);
            border-radius: 10px;
            padding: 1rem;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(10px);
        }
        
        [data-testid="metric-container"] [data-testid="metric-value"] {
            color: #ffffff;
            font-weight: 700;
        }
        
        [data-testid="metric-container"] [data-testid="metric-label"] {
            color: #b0b0b0;
        }
        
        /* Tab styling - Dark theme */
        .stTabs [data-baseweb="tab-list"] {
            background: rgba(30, 35, 50, 0.8);
            border-radius: 8px;
            padding: 0.5rem;
            backdrop-filter: blur(10px);
        }
        
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            border-radius: 6px;
            color: #b0b0b0;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
        }
        
        /* Expander styling - Dark theme */
        .streamlit-expanderHeader {
            background: rgba(30, 35, 50, 0.9);
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            color: #ffffff;
        }
        
        /* Data frame styling - Dark theme */
        .dataframe {
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            background: rgba(30, 35, 50, 0.9);
            backdrop-filter: blur(10px);
        }
        
        /* Progress bar - Dark theme */
        .stProgress > div > div > div {
            background: linear-gradient(45deg, #667eea, #764ba2);
        }
        
        /* File uploader styling - Dark theme */
        .stFileUploader {
            background: rgba(30, 35, 50, 0.9);
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
        }
        
        /* Download button styling - Dark theme */
        .stDownloadButton > button {
            background: rgba(30, 35, 50, 0.9);
            border: 2px solid #667eea;
            color: #667eea;
            border-radius: 8px;
            transition: all 0.3s ease;
            backdrop-filter: blur(10px);
        }
        
        .stDownloadButton > button:hover {
            background: #667eea;
            color: white;
            transform: translateY(-1px);
        }
        
        /* Success/Error messages - Dark theme */
        .stSuccess {
            background: rgba(72, 187, 120, 0.2);
            border: 1px solid rgba(72, 187, 120, 0.4);
            border-radius: 8px;
            backdrop-filter: blur(10px);
            color: #90f0a8;
        }
        
        .stError {
            background: rgba(245, 101, 101, 0.2);
            border: 1px solid rgba(245, 101, 101, 0.4);
            border-radius: 8px;
            backdrop-filter: blur(10px);
            color: #ffa8a8;
        }
        
        /* Spinner styling - Dark theme */
        .stSpinner {
            color: #667eea;
        }
        
        /* Text colors - Dark theme */
        .stMarkdown {
            color: #ffffff;
        }
        
        /* Slider styling - Dark theme */
        .stSlider > div > div > div {
            background: linear-gradient(45deg, #667eea, #764ba2);
        }
        
        /* Selectbox dropdown - Dark theme */
        .stSelectbox > div > div > div {
            background: rgba(30, 35, 50, 0.9);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        /* Text area - Dark theme */
        .stTextArea > div > div > textarea {
            background: rgba(30, 35, 50, 0.9);
            border: 2px solid rgba(255, 255, 255, 0.1);
            color: #ffffff;
        }
        
        /* Sidebar text - Dark theme */
        .css-1d391kg {
            color: #ffffff;
        }
        
        /* Override any light theme remnants */
        .stApp > header {
            background: transparent;
        }
        
        .stApp .main .block-container {
            background: rgba(20, 25, 40, 0.95);
        }
        
        /* Force dark scrollbars */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(30, 35, 50, 0.5);
        }
        
        ::-webkit-scrollbar-thumb {
            background: rgba(102, 126, 234, 0.6);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(102, 126, 234, 0.8);
        }
        
        /* Remove default spacing */
        .element-container {
            margin: 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Professional title section
    st.markdown('<div class="header-section">', unsafe_allow_html=True)
    st.markdown('<h1 class="main-title">Advanced Logical Fallacy Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Professional AI-powered analysis with multiple modes and credibility assessment</p>', unsafe_allow_html=True)
    
    # Add portfolio context
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Analysis Modes", "3", help="Single, Multiple, Batch")
    with col2:
        st.metric("Fallacy Types", "13+", help="Based on Aristotelian logic")
    with col3:
        st.metric("AI Models", "3", help="GPT-4o, GPT-4o-mini, GPT-3.5-turbo")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    #  info
    with st.sidebar.expander("About This Project"):
        st.markdown("""
        **Built with:**
        - Python & Streamlit
        - LangChain & OpenAI
        - Plotly for visualizations
        - Pandas for data handling
        
        **Features:**
        - Multi-modal analysis
        - Batch processing
        - Real-time visualizations
        - Export capabilities
        
        **Use Cases:**
        - Media literacy education
        - Journalism fact-checking
        - Academic research
        - Content quality assessment
        """)
    
    # API Keys
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    serper_api_key = st.sidebar.text_input("Serper API Key", type="password")
    
    # Analysis mode
    analysis_mode = st.sidebar.selectbox(
        "Analysis Mode",
        ["Single Article Analysis", "Multiple Article Comparison", "Batch Analysis"]
    )
    
    if analysis_mode == "Multiple Article Comparison":
        num_articles = st.sidebar.slider("Number of Articles", 2, 5, 3)
    
    # Model settings
    model_choice = st.sidebar.selectbox(
        "Model",
        ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
    )
    
    #  footer
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<div style='text-align: center; color: #f797e1; font-size: 0.8em;'>"
        "© 2025 A.R."
        "</div>", 
        unsafe_allow_html=True
    )
    
    # Initialize analyzer
    if openai_api_key and serper_api_key:
        analyzer = FallacyAnalyzer(openai_api_key, serper_api_key, model_choice)
        fallacies_df = analyzer.load_fallacies()
        
        if fallacies_df is not None:
            
            # Display fallacies reference
            with st.expander("Logical Fallacies Reference"):
                st.dataframe(fallacies_df)
            
            # Main analysis interface
            if analysis_mode == "Single Article Analysis":
                single_article_interface(analyzer)
            
            elif analysis_mode == "Multiple Article Comparison":
                multiple_article_interface(analyzer, num_articles)
            
            elif analysis_mode == "Batch Analysis":
                batch_analysis_interface(analyzer)
    
    else:
        st.warning("Please enter your API keys to begin analysis.")

def single_article_interface(analyzer):
    """Interface for single article analysis"""
    
    st.subheader("Single Article Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        search_topic = st.text_input("Search Topic", placeholder="e.g., climate change policy")
        
        if st.button("Analyze Article", type="primary"):
            if search_topic:
                with st.spinner("Analyzing article..."):
                    try:
                        # Get first article
                        search = GoogleSerperAPIWrapper(
                            type="news",
                            tbs="qdr:m1",
                            serper_api_key=analyzer.serper_api_key
                        )
                        
                        search_results = search.results(search_topic)
                        first_article = search_results['news'][0]
                        
                        result = analyzer.analyze_single_article(
                            first_article['link'], 
                            first_article['title']
                        )
                        
                        # Display results
                        st.success("Analysis completed!")
                        
                        # Article info
                        st.markdown("### Article Information")
                        st.markdown(f"**Title:** {result['title']}")
                        st.markdown(f"**URL:** [{result['url']}]({result['url']})")
                        
                        # Analysis tabs
                        tab1, tab2, tab3 = st.tabs(["Summary", "Fallacy Analysis", "Credibility Assessment"])
                        
                        with tab1:
                            st.markdown(result['summary'])
                        
                        with tab2:
                            st.markdown(result['fallacy_analysis'])
                        
                        with tab3:
                            st.markdown(result['credibility_analysis'])
                        
                        # Export
                        json_data = json.dumps(result, indent=2)
                        st.download_button(
                            "Download Analysis",
                            json_data,
                            f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            "application/json"
                        )
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    with col2:
        st.markdown("### Analysis Features")
        st.markdown("""
        - Detailed article summarization
        - Multiple fallacy detection
        - Confidence scoring
        - Credibility assessment
        - Alternative perspectives
        - Improvement recommendations
        """)

def multiple_article_interface(analyzer, num_articles):
    """Interface for multiple article comparison"""
    
    st.subheader("Multiple Article Comparison")
    
    search_topic = st.text_input("Search Topic", placeholder="e.g., artificial intelligence regulation")
    
    if st.button("Compare Articles", type="primary"):
        if search_topic:
            with st.spinner(f"Analyzing {num_articles} articles..."):
                try:
                    results = analyzer.analyze_multiple_articles(search_topic, num_articles)
                    
                    if results:
                        # Create visualization
                        fig = analyzer.create_visualization(results)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Display results
                        for i, result in enumerate(results):
                            with st.expander(f"Article {i+1}: {result['title'][:60]}..."):
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("**Fallacy Analysis:**")
                                    st.markdown(result['fallacy_analysis'])
                                
                                with col2:
                                    st.markdown("**Credibility Assessment:**")
                                    st.markdown(result['credibility_analysis'])
                                
                                st.markdown(f"**URL:** [{result['url']}]({result['url']})")
                        
                        # Export all results
                        json_data = json.dumps(results, indent=2)
                        st.download_button(
                            "Download All Analyses",
                            json_data,
                            f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            "application/json"
                        )
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")

def batch_analysis_interface(analyzer):
    """Interface for batch analysis"""
    
    st.subheader("Batch Analysis")
    
    st.markdown("Upload a file with URLs or search topics (one per line)")
    
    uploaded_file = st.file_uploader("Choose a text file", type="txt")
    
    if uploaded_file is not None:
        content = str(uploaded_file.read(), "utf-8")
        topics = [line.strip() for line in content.split('\n') if line.strip()]
        
        st.write(f"Found {len(topics)} topics/URLs")
        
        if st.button("Start Batch Analysis", type="primary"):
            progress_bar = st.progress(0)
            results = []
            
            for i, topic in enumerate(topics):
                try:
                    with st.spinner(f"Analyzing {i+1}/{len(topics)}: {topic[:50]}..."):
                        if topic.startswith('http'):
                            # Direct URL analysis
                            result = analyzer.analyze_single_article(topic, f"Article {i+1}")
                        else:
                            # Search topic analysis
                            search = GoogleSerperAPIWrapper(
                                type="news",
                                tbs="qdr:m1",
                                serper_api_key=analyzer.serper_api_key
                            )
                            search_results = search.results(topic)
                            first_article = search_results['news'][0]
                            result = analyzer.analyze_single_article(
                                first_article['link'], 
                                first_article['title']
                            )
                        
                        results.append(result)
                        progress_bar.progress((i + 1) / len(topics))
                        
                except Exception as e:
                    st.warning(f"Failed to analyze: {topic} - {str(e)}")
                    continue
            
            # Display summary
            st.success(f"Batch analysis completed! Analyzed {len(results)} articles.")
            
            # Create summary visualization
            if results:
                fig = analyzer.create_visualization(results)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            # Export results
            json_data = json.dumps(results, indent=2)
            st.download_button(
                "Download Batch Results",
                json_data,
                f"batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "application/json"
            )

if __name__ == "__main__":
    main()

print("Current working directory:", os.getcwd())
print("Attempting to load:", os.path.abspath('../data/fallacies.csv'))
