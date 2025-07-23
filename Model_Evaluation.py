import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import torch.nn.functional as F
from tqdm import tqdm
import os



# ===== Dataset for Text Prompts =====
class PromptDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        prompt = self.dataframe.iloc[idx]['prompt']
        label = self.dataframe.iloc[idx]['label']
        
        encoding = self.tokenizer(prompt,
                                  padding='max_length',
                                  truncation=True,
                                  max_length=self.max_length,
                                  return_tensors='pt')

        # Squeeze to remove the batch dimension
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label, dtype=torch.long)
        }

# ===== Evaluation loop =====
def evaluate_model(model, dataloader, device, test_path):
    model.eval()
    y_pred = []
    y_true = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", unit="batch"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(F.softmax(outputs.logits.float(), dim=-1), dim=1)
            
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    # Save predictions to CSV
    df_test = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    df_test.to_csv(test_path, index=False)
    print(f"Predictions saved to {test_path}")

    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    # Print metrics
    print(f"\nAccuracy: {accuracy * 100:.2f}%")
    print(f"F1 Score: {f1:.2f}")
    print("Confusion Matrix:")
    print(cm)

    # Plot confusion matrix as a heatmap with a blue-green color palette
    plt.figure(figsize=(6, 5))
    heatmap = sns.heatmap(cm, annot=True, fmt='d', xticklabels=['Image', 'Text'], yticklabels=['Image', 'Text'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Save the heatmap figure
    figure = heatmap.get_figure()
    figure.savefig('confusion_matrix.png', bbox_inches='tight')

# ===== Main entry point =====
def main():
        
    # ====== DEVICE CHECK ======
    print("CUDA Available:", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    # Get the directory where this script is located
    project_dir = os.path.dirname(os.path.abspath(__file__))

    # Load the test CSV

    test_csv_path = os.path.join(project_dir, "datasets", r"test_dataset.csv")
    test_df = pd.read_csv(test_csv_path)

    # Convert class labels to binary (1 = text, 0 = image)
    test_df['label'] = test_df['class'].apply(lambda x: 1 if x.lower() == 'text' else 0)

    # Construct the relative model path
    model_path = os.path.join(project_dir, "models", r"roBERTa_Large_Prompt_Classification_Model")
    print (f"model_path: {model_path}\n\n\n\n")

    # ====== MODEL LOADING ======
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("Model and tokenizer loaded successfully.")

    # Dataset and DataLoader
    test_dataset = PromptDataset(test_df, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8)

    # Evaluate and generate predictions.csv
    test_path = 'predictions.csv'
    evaluate_model(model, test_loader, device, test_path)

# ===== Windows-safe script guard =====
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
