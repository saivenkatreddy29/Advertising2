from transformers import pipeline

# Initialize the zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def classify_with_non_relevant(text, labels):
    # Add 'non-relevant' to the list of labels
    all_labels = labels + ['non-relevant']
    
    # Perform zero-shot classification
    result = classifier(text, all_labels, multi_label=True)
    print(f'result is :{result}')
    
    # Get the top predicted label and its score
    top_label = result['labels'][0]
    top_score = result['scores'][0]
    
    # If the top label is not 'non-relevant' and its score is above a threshold, return it
    # Otherwise, return 'non-relevant'
    threshold = 0.5  # You can adjust this threshold as needed
    if top_label != 'non-relevant' and top_score > threshold:
        return top_label
    else:
        return 'non-relevant'

# Example usage
text = "The stock market showed significant growth this quarter."
text2 = "Surya is a very good player and also good in finance"
text3 = 'I invented a Rocket and I have generate good revenue'
text4 = 'I am feeling sick today'
labels = ['finance', 'sports', 'technology', 'entertainment', 'politics']

result = classify_with_non_relevant(text3, labels)
print(f"Classified as: {result}")