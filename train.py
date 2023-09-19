import torch
from transformers import AutoModelForSequenceClassification, AdamW, AutoTokenizer, AutoModelForMultipleChoice, DataCollatorWithPadding
from torch.utils.data import DataLoader
from data import create_dataset


def train(model, train_loader, optimizer, device):
    optimizer.zero_grad()
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        optimizer.zero_grad()

        inputs = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**inputs)
        loss = outputs[0]

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    return avg_loss

def main():
    base_dir = "data/mutual"
    tokenizer_name = 'roberta-base' # debug
    model_name = 'roberta-base' # debug
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    max_seq_length = 256
    batch_size = 2 # debug
    learning_rate = 2e-5
    epochs = 3

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = create_dataset(base_dir, 'train', tokenizer, max_seq_length)
    collate_fn = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn = collate_fn)
    num_labels = 2  
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = num_labels)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        avg_loss = train(model, train_loader, optimizer, device)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.3f}")

    # Save the model
    model.save_pretrained('./mutual_model')

if __name__ == "__main__":
    main()
