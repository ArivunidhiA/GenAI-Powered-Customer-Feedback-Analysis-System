# GenAI-Powered Customer Feedback Analysis System

## Overview
This project implements a sophisticated customer feedback analysis system using OpenAI's GPT models. It processes customer reviews to extract deep insights, sentiment patterns, and actionable recommendations using artificial intelligence. The system uses real Amazon Electronics review data combined with GPT's advanced natural language understanding to provide strategic insights for product improvement.

## Key Features
- AI-powered sentiment analysis using GPT
- Intelligent topic extraction and categorization
- Automated improvement recommendations
- Interactive visualizations using Plotly
- Comprehensive AI-generated insights report
- Strategic recommendation generation

## Technical Features
- Uses GPT-3.5-turbo for advanced text analysis
- Batch processing with rate limiting
- JSON-structured AI responses
- Interactive data visualizations
- Automated report generation

## Requirements
```
pandas
numpy
plotly
matplotlib
wordcloud
openai
tqdm
```

## Installation
1. Clone this repository:
```bash
git clone https://github.com/yourusername/genai-feedback-analysis.git
cd genai-feedback-analysis
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key:
- Create an account at OpenAI
- Get your API key
- Replace 'your-api-key-here' in the main() function with your actual API key

## Usage
Run the main script:
```bash
python feedback_analyzer.py
```

The system will:
1. Load the review dataset
2. Perform AI analysis using GPT
3. Generate interactive visualizations
4. Create detailed insight reports
5. Provide strategic recommendations

## Output Files
The system generates several files:
- `gpt_sentiment_distribution.html`: Interactive pie chart of AI-analyzed sentiments
- `gpt_topic_distribution.html`: Bar chart of AI-identified topics
- `ai_insights.json`: Detailed JSON report of AI findings
- `ai_recommendations.md`: Strategic recommendations based on AI analysis

## How It Works
1. **Data Processing**: Loads and prepares customer review data
2. **AI Analysis**: Uses GPT to analyze each review for:
   - Sentiment classification
   - Topic identification
   - Improvement suggestions
3. **Insight Generation**: Aggregates AI analyses to identify patterns and trends
4. **Recommendation Engine**: Generates strategic recommendations based on AI findings
5. **Visualization**: Creates interactive charts and reports

## AI Capabilities
- **Sentiment Analysis**: Advanced u