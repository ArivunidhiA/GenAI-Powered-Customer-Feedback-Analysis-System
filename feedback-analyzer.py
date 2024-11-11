import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import openai
import json
from datetime import datetime
import time
from tqdm import tqdm

class FeedbackAnalyzer:
    def __init__(self, api_key):
        # Initialize OpenAI client
        openai.api_key = api_key
        
        # Load the dataset
        self.df = pd.read_csv('https://raw.githubusercontent.com/HaoranYu1994/Amazon-Product-Review-Dataset/master/Electronics_5.csv')
        self.df = self.df.head(1000)  # Using 1000 reviews for demonstration
        self.process_data()
        
    def process_data(self):
        # Clean and prepare data
        self.df['reviewTime'] = pd.to_datetime(self.df['reviewTime'])
        self.df['year'] = self.df['reviewTime'].dt.year
        self.df['month'] = self.df['reviewTime'].dt.month
        
        # Initialize columns for GPT analysis
        self.df['gpt_sentiment'] = None
        self.df['gpt_topics'] = None
        self.df['gpt_suggestions'] = None

    def analyze_with_gpt(self, review_text):
        """Use GPT to analyze a single review"""
        try:
            prompt = f"""Analyze this product review and provide the following in JSON format:
            1. sentiment: (positive, negative, or neutral)
            2. topics: (list of main topics discussed)
            3. improvement_suggestions: (list of actionable suggestions based on the review)

            Review: {review_text}

            Return only valid JSON without explanation."""

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a product analysis expert. Analyze the review and return only JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )

            # Parse the JSON response
            analysis = json.loads(response.choices[0].message.content)
            return analysis
            
        except Exception as e:
            print(f"Error analyzing review: {e}")
            return {
                "sentiment": "neutral",
                "topics": ["error"],
                "improvement_suggestions": ["Unable to analyze"]
            }

    def batch_analyze_reviews(self):
        """Process all reviews with GPT"""
        print("Analyzing reviews with GPT...")
        
        for idx in tqdm(range(len(self.df))):
            review = self.df.iloc[idx]['reviewText']
            
            # Skip if review is too short or NaN
            if pd.isna(review) or len(str(review)) < 10:
                continue
                
            analysis = self.analyze_with_gpt(str(review))
            
            # Store results
            self.df.at[idx, 'gpt_sentiment'] = analysis['sentiment']
            self.df.at[idx, 'gpt_topics'] = json.dumps(analysis['topics'])
            self.df.at[idx, 'gpt_suggestions'] = json.dumps(analysis['improvement_suggestions'])
            
            # Sleep to respect API rate limits
            time.sleep(0.5)

    def generate_insights(self):
        """Generate visualizations and insights from GPT analysis"""
        # 1. GPT Sentiment Distribution
        sentiment_dist = self.df['gpt_sentiment'].value_counts()
        fig_sentiment = px.pie(
            values=sentiment_dist.values,
            names=sentiment_dist.index,
            title='AI-Analyzed Sentiment Distribution',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_sentiment.write_html("gpt_sentiment_distribution.html")
        
        # 2. Topic Analysis
        all_topics = []
        for topics_str in self.df['gpt_topics'].dropna():
            all_topics.extend(json.loads(topics_str))
        
        topic_counts = Counter(all_topics)
        fig_topics = px.bar(
            x=list(topic_counts.keys()),
            y=list(topic_counts.values()),
            title='AI-Identified Topics in Reviews',
            labels={'x': 'Topic', 'y': 'Count'},
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_topics.write_html("gpt_topic_distribution.html")
        
        # 3. Generate Key Insights and Recommendations Report
        all_suggestions = []
        for sugg_str in self.df['gpt_suggestions'].dropna():
            all_suggestions.extend(json.loads(sugg_str))
            
        common_suggestions = Counter(all_suggestions).most_common(10)
        
        # Prepare insights dictionary
        insights = {
            'total_reviews_analyzed': len(self.df),
            'sentiment_distribution': dict(sentiment_dist),
            'top_topics': dict(topic_counts.most_common(5)),
            'top_suggestions': dict(common_suggestions),
            'analysis_date': datetime.now().strftime("%Y-%m-%d")
        }
        
        # Save insights to JSON
        with open('ai_insights.json', 'w') as f:
            json.dump(insights, f, indent=4)
            
        return insights

    def generate_recommendation_report(self):
        """Generate a detailed recommendation report based on GPT analysis"""
        recommendations = []
        
        # Group suggestions by sentiment
        neg_reviews = self.df[self.df['gpt_sentiment'] == 'negative']
        for _, row in neg_reviews.iterrows():
            if pd.notna(row['gpt_suggestions']):
                suggestions = json.loads(row['gpt_suggestions'])
                recommendations.extend(suggestions)
        
        # Count and sort recommendations
        recommendation_counts = Counter(recommendations)
        
        # Generate report
        report = "# AI-Generated Product Improvement Recommendations\n\n"
        report += f"Analysis Date: {datetime.now().strftime('%Y-%m-%d')}\n\n"
        report += "## Top Recommendations\n\n"
        
        for rec, count in recommendation_counts.most_common(10):
            report += f"- {rec} (mentioned {count} times)\n"
            
        # Save report
        with open('ai_recommendations.md', 'w') as f:
            f.write(report)
            
        return report

def main():
    # Initialize with your OpenAI API key
    api_key = "your-api-key-here"  # Replace with actual API key
    
    print("Initializing AI-Powered Feedback Analyzer...")
    analyzer = FeedbackAnalyzer(api_key)
    
    print("Performing AI analysis of reviews...")
    analyzer.batch_analyze_reviews()
    
    print("Generating AI insights...")
    insights = analyzer.generate_insights()
    
    print("Generating AI recommendations...")
    recommendations = analyzer.generate_recommendation_report()
    
    print("\nAnalysis Complete! Key Insights:")
    print(f"Total Reviews Analyzed: {insights['total_reviews_analyzed']}")
    print(f"Sentiment Distribution: {insights['sentiment_distribution']}")
    print("\nTop Topics Identified by AI:")
    for topic, count in insights['top_topics'].items():
        print(f"- {topic}: {count}")
    
    print("\nFiles generated:")
    print("- gpt_sentiment_distribution.html")
    print("- gpt_topic_distribution.html")
    print("- ai_insights.json")
    print("- ai_recommendations.md")

if __name__ == "__main__":
    main()
