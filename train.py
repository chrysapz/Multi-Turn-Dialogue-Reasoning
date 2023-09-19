import torch
from transformers import RobertaForMultipleChoice, AdamW, AutoTokenizer, AutoModelForMultipleChoice
from torch.utils.data import DataLoader
from data import create_dataset

def train(model, train_loader, optimizer, device):
    optimizer.zero_grad()
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        optimizer.zero_grad()

        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[2]}
        outputs = model(**inputs)
        loss = outputs[0]

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    return avg_loss

def main():
    base_dir = "data/mutual"
    tokenizer_name = 'roberta-large'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    max_seq_length = 256
    batch_size = 16
    learning_rate = 2e-5
    epochs = 3

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = create_dataset(base_dir, 'train', tokenizer, max_seq_length)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

    num_labels = 2  
    model = AutoModelForMultipleChoice.from_pretrained(tokenizer_name, num_labels=num_labels)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        avg_loss = train(model, train_loader, optimizer, device)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.3f}")

    # Save the model
    model.save_pretrained('./mutual_model')

if __name__ == "__main__":
    main()
