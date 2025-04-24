from transformers import pipeline

def classify_intent(text):
    """
    Classify text as intent or keyword based.
    
    Args:
        text (str): The input text to classify
        
    Returns:
        list: Classification results with label and score
    """
    # Load the pre-trained model for text classification
    pipe = pipeline("text-classification", model="roundspecs/minilm-finetuned-intent-classification")
    
    # Get the classification result
    result = pipe(text)
    
    return result

if __name__ == "__main__":
    # Example input text
    text = "gaming pc under 50000"
    
    # Get the classification result
    result = classify_intent(text)
    
    # Print the result
    print(result)
