from transformers import pipeline

# Initialize the zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def classify_with_multiple_labels(text, labels, threshold=0.3):
    """
    Classify text with support for multiple relevant labels.
    
    Args:
        text (str): Input text to classify
        labels (list): List of possible labels
        threshold (float): Confidence threshold for accepting a label (default: 0.3)
    
    Returns:
        list: List of tuples containing (label, score) for all relevant labels
    """
    # Add 'non-relevant' to the list of labels
    all_labels = labels + ['non-relevant']
    
    # Perform zero-shot classification
    result = classifier(text, all_labels, multi_label=True)
    
    # Create list of (label, score) tuples
    label_scores = list(zip(result['labels'], result['scores']))
    
    # Filter labels that meet the threshold and aren't 'non-relevant'
    relevant_labels = [
        (label, score) 
        for label, score in label_scores 
        if score > threshold and label != 'non-relevant'
    ]
    
    # Sort by score in descending order
    relevant_labels.sort(key=lambda x: x[1], reverse=True)
    
    # If no labels meet the criteria, return 'non-relevant'
    if not relevant_labels:
        return [('non-relevant', max(score for label, score in label_scores if label == 'non-relevant'))]
    
    return relevant_labels

def print_classification_results(text, result):
    """
    Pretty print the classification results.
    """
    print(f"\nText: {text}")
    print("Classifications:")
    for label, score in result:
        print(f"- {label}: {score:.3f}")
    print("-" * 50)

# Example usage
text_examples = [
    "The stock market showed significant growth this quarter, while tech companies reported record profits.",
    "Surya scored a century in cricket and also gave financial advice to his teammates.",
    "I invented a rocket that generates significant revenue from satellite launches.",
    "I am feeling sick today.",
]

labels = ['finance', 'sports', 'technology', 'entertainment', 'politics']

# Test with multiple examples
for text in text_examples:
    results = classify_with_multiple_labels(text, labels)
    print_classification_results(text, results)