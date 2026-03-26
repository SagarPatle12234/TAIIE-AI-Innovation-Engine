import pandas as pd
import re

def clean_topic(topic):
    topic = topic.lower()
    topic = re.sub(r'[^a-z0-9\s]', '', topic)
    topic = re.sub(r'\s+', ' ', topic).strip()
    return topic

def check_topics_in_csv(topics, csv_path):
    """Check multiple topics at once"""
    df = pd.read_csv(csv_path)
    df['cleaned_topic'] = df['Research Topic'].apply(clean_topic)
    
    results = {}
    for topic in topics:
        cleaned = clean_topic(topic)
        exists = cleaned in df['cleaned_topic'].values
        original = df[df['cleaned_topic'] == cleaned]['Research Topic'].values[0] if exists else None
        results[topic] = {'exists': exists, 'original': original}
    
    return results

# Example usage
if __name__ == "__main__":
    topics_to_check = [
        "bayesian classification methods",
        "bayesian mining for particle physics applications",
        "bayesian inference methods",
        "bayesian reconstruction for computer vision applications",
        "bayesian linear regression",
        "bayesian forecasting for ecommerce applications",
        "bayesian prediction in energy",
        "Sarcasm detection in news media",
        "bayesian quantum error correction systems",
        "bayesian quantum decomposition using machine learning",
        "analysis for visually observational economics",
        " independent component analysis"
    ]
    
    results = check_topics_in_csv(topics_to_check, "preexisting_research_topics_cleaned.csv")
    
    for topic, data in results.items():
        status = "FOUND" if data['exists'] else "NOT FOUND"
        original = f"({data['original']})" if data['exists'] else ""
        print(f"{topic}: {status} {original}")