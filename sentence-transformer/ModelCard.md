# Model Card: DistilBERT SST-2 Fine-Tuned

## Model Overview

The `distilbert/distilbert-base-uncased-finetuned-sst-2-english` model is a Transformer-based DistilBERT variant fine-tuned for sentiment analysis. It is sourced from the Hugging Face Model Hub and licensed under Apache-2.0. This model efficiently classifies text as positive or negative, making it suitable for various NLP applications requiring sentiment detection.

## Model Description

This model is a distilled version of BERT, fine-tuned on the Stanford Sentiment Treebank (SST-2) dataset. Distillation reduces the model size by 60% while retaining 97% of BERT's original performance, enabling faster inference. It is designed for sentiment classification and provides a balance between accuracy and efficiency, making it ideal for real-time applications.

## Intended Use

The model is designed for real-time and batch sentiment analysis in customer feedback processing, social media monitoring, chatbot interactions, and enterprise sentiment workflows. It is best suited for general English text and works well with structured reviews and conversational text. While effective in many applications, it is not optimized for highly domain-specific language, such as medical or legal text.

## Input and Output Format

The model accepts English text as input, which must be tokenized using the `distilbert-base-uncased` tokenizer. The output consists of a sentiment label, either positive or negative, along with a confidence score indicating the model’s certainty in its classification.

## Performance

The model achieves approximately 91.3% accuracy on the SST-2 benchmark dataset. It provides fast inference times while maintaining strong performance in structured sentiment classification tasks. In addition to accuracy, it performs well on other key evaluation metrics, including precision, recall, and F1-score. However, performance may degrade when analyzing short, ambiguous, or sarcastic text.

## Limitations and Considerations

This model only supports binary sentiment classification and does not recognize neutral or mixed sentiment. It may struggle with sarcasm, complex negation, and domain-specific jargon. Additionally, since it was trained on movie reviews, biases from the training data may affect its predictions on other types of content. To improve robustness, users may consider additional fine-tuning or post-processing techniques.

## Deployment in JFrog ML

This model can be deployed in JFrog ML for enterprise sentiment analysis. It can be integrated into workflows for real-time sentiment tracking, customer feedback analysis, and chatbot sentiment detection. JFrog ML provides capabilities for secure model management, performance monitoring, and compliance with governance policies, ensuring reliable deployment in production environments.

