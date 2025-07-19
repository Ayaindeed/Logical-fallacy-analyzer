# Logical Fallacy News Analyzer

An AI-powered tool that analyzes news articles for logical fallacies, built with Streamlit and LangChain.

## Features

- Single article analysis
- Multiple article comparison (2-5 articles)
- Batch analysis from file upload
- Logical fallacy detection
- Credibility assessment
- Confidence scoring
- Interactive visualizations
- Export results as JSON

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Get API Keys:**
   - [OpenAI API Key](https://platform.openai.com/api-keys)
   - [Serper API Key](https://serper.dev/)

3. **Run the application:**
   ```bash
   streamlit run advanced_streamlit_app.py
   ```

## Usage

1. Enter your API keys in the sidebar
2. Select analysis mode (Single/Multiple/Batch)
3. Enter search topic or upload file
4. Review results and export if needed


## Supported Fallacies
- Ad Hominem, Appeal to Emotion, False Dilemma
- Hasty Generalization, False Causality
- Slippery Slope, Appeal to Authority
- And more (see fallacies.csv for complete list)

## Technical Stack

- **Frontend**: Streamlit
- **AI**: OpenAI GPT models, LangChain
- **Search**: Serper API
- **Data**: Pandas, Plotly
- **Models**: GPT-4o-mini, GPT-4o, GPT-3.5-turbo
