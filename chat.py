import torch
import torch.nn as nn
import json

# Model class
class ChatBot(nn.Module):
    def __init__(self, model_path, vocab_path):
        super(ChatBot, self).__init__()
        self.model_path = model_path
        self.vocab_path = vocab_path
        self.load_model()
        self.load_vocab()

    def load_model(self):
        # Load model from the specified path
        checkpoint = torch.load(self.model_path, map_location=torch.device('cpu'))
        self.model = checkpoint['model']
        self.model.eval()

    def load_vocab(self):
        # Load vocabulary from the specified path
        with open(self.vocab_path, 'r') as f:
            self.vocab = json.load(f)
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

    def encode_text(self, text):
        tokens = text.split()
        ids = [self.vocab.get(token, self.vocab['[UNK]']) for token in tokens]
        return torch.tensor(ids).unsqueeze(0)

    def decode_ids(self, ids):
        tokens = [self.reverse_vocab.get(id, '[UNK]') for id in ids]
        return ' '.join(tokens)

    def generate_response(self, input_text):
        input_tensor = self.encode_text(input_text)
        with torch.no_grad():
            output = self.model(input_tensor)
            predicted_ids = torch.argmax(output, dim=-1).squeeze().tolist()
            response = self.decode_ids(predicted_ids)
        return response

# Main function to run the chatbot
def main():
    model_path = "path/to/model.nit"
    vocab_path = "path/to/vocab.json"

    chatbot = ChatBot(model_path, vocab_path)
    print("Chatbot tayyor. Suhbatni yopish uchun 'exit' deb yozing.")

    while True:
        user_input = input("Siz: ")
        if user_input.lower() == 'exit':
            print("Chatbotni yopish...")
            break
        
        response = chatbot.generate_response(user_input)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main()
