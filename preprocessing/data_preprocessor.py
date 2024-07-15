import re
import numpy as np
from transformers import BertTokenizer

class DataPreprocessor:
    def __init__(self, text_data: list, bert_model_name='bert-base-uncased'):
        self.text_data = text_data
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)

    def parse_annotations(self) -> list:
        # Extract annotated entities and their labels
        pattern = re.compile(r"\{\{(.*?)\}\}\{\{(.*?)\}\}")
        annotations = []
        for text in self.text_data:
            matches = pattern.findall(text)
            annotations.extend(matches)
        return annotations

    def tokenize_text(self, text: str) -> list:
        # Use BERT tokenizer to tokenize text while preserving named entities
        tokens = []
        pattern = re.compile(r"\{\{.*?\}\}\{\{.*?\}\}|{{CRLF}}|[^\s]+")
        matches = pattern.findall(text)
        for match in matches:
            if match.startswith('{{') and match.endswith('}}'):
                tokens.append(match)
            elif match == '{{CRLF}}':
                tokens.append(match)
            else:
                tokens.extend(self.tokenizer.tokenize(match))
        return tokens

    def generate_training_samples(self) -> tuple:
        annotations = self.parse_annotations()
        X, y = [], []
        identified_entities = []
        for text in self.text_data:
            lines = text.split('{{CRLF}}')
            for line_num, line in enumerate(lines):
                tokens = self.tokenize_text(line)
                token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                for pos, token in enumerate(tokens):
                    if token.startswith('{{') and token.endswith('}}'):
                        entity, label = self.extract_entity_label(token)
                        identified_entities.append((entity, label))
                        continue  # Skip named entities
                    feature = {
                        'token': token,
                        'token_id': token_ids[pos],
                        'x': pos,
                        'y': line_num
                    }
                    X.append(feature)
                    y.append('O')  # All tokens are labeled as 'O'
        return np.array(X), np.array(y), identified_entities

    def extract_entity_label(self, token: str) -> tuple:
        # Extract the entity and its label from the token
        pattern = re.compile(r"\{\{(.*?)\}\}\{\{(.*?)\}\}")
        match = pattern.match(token)
        if match:
            return match.group(1), match.group(2)
        return None, None

# Example usage
text_data = [
    "This is a sample text {{Robert Cobbins}}{{PERSON_NAME}} {{CRLF}} New line here",
    "Another line with {{Apple}}{{ORG_NAME}} and some text {{CRLF}} More text"
]

preprocessor = DataPreprocessor(text_data)
X, y, identified_entities = preprocessor.generate_training_samples()

print("Preprocessed Data")
print("Tokens and Labels:")
for feature, label in zip(X, y):
    print(f"Token: {feature['token']}, Token ID: {feature['token_id']}, x: {feature['x']}, y: {feature['y']}, Label: {label}")

print("\nIdentified Entities:")
for entity, label in identified_entities:
    print(f"Entity: {entity}, Label: {label}")

