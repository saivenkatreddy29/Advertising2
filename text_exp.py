from transformers import pipeline

# Initialize the zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def classify_with_multiple_labels(text, labels, threshold=0.5):
    # Add 'non-relevant' to the list of labels
    all_labels = labels + ['non-relevant']
    
    # Perform zero-shot classification with multi_label=True
    result = classifier(text, all_labels, multi_label=True)
    print(f'result:{result}')
    # Collect labels that have scores above the threshold and are not 'non-relevant'
    selected_labels = [
        label for label, score in zip(result['labels'], result['scores'])
        if label != 'non-relevant' and score > threshold
    ]
    
    # If no labels are above the threshold, return 'non-relevant'
    if not selected_labels:
        return ['non-relevant']
    return selected_labels

# Example usage
text = "The stock market showed significant growth this quarter."
text2 = "Surya is a very good player and also good in finance"
text3 = 'I invented a Rocket and I have generated good revenue'
text4 = 'I am feeling sick today'
labels = ['finance', 'sports', 'technology', 'entertainment', 'politics']

result = classify_with_multiple_labels(text2, labels)
print(f"Classified as: {result}")
