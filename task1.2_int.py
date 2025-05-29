# 取整数
import sys
import jsonlines
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 本地加载保存的最佳模型
# print("加载保存的最佳模型")
# best_model_path = "saved_model/best_model3.pth"
# tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
# model = AutoModelForSequenceClassification.from_pretrained("answerdotai/ModernBERT-base", num_labels=2).to(device)
# print("模型加载完成")
# model.load_state_dict(torch.load(best_model_path, map_location=device))

# 使用huggingface上传的模型
tokenizer = AutoTokenizer.from_pretrained("kkkkl5/asdkklkk")
model = AutoModelForSequenceClassification.from_pretrained("kkkkl5/asdkklkk", num_labels=2).to(device)

model.eval()

# 定义自定义数据集类用于测试数据
class TestDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        with jsonlines.open(file_path, 'r') as reader:
            for item in reader:
                text = item['text']
                id = item['id']
                inputs = tokenizer(text, truncation=True, padding='max_length', max_length=512)
                self.data.append({
                    'id': id,
                    'input_ids': torch.tensor(inputs['input_ids']),
                    'attention_mask': torch.tensor(inputs['attention_mask'])
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# 定义预测函数
def predict(model, dataloader, output_dir):
    model.eval()
    results = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting", unit="batch"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            ids = batch['id']

            outputs = model(input_ids, attention_mask=attention_mask)
            confidence_scores = torch.sigmoid(outputs.logits[:, 1])  # 获取为 AI 写作的置信度并应用 sigmoid

            # 将置信度分数转换为二进制标签
            binary_labels = (confidence_scores >= 0.5).int()  # 直接转换为整数类型

            for id, label in zip(ids, binary_labels):
                results.append({
                    "id": id,
                    "label": label.item()  # 保存整数类型的二进制标签
                })

    # 保存结果到 JSONL 文件
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, "new_predictions2_int.jsonl")
    with jsonlines.open(output_file_path, mode='w') as writer:
        for item in results:
            writer.write(item)

# 获取命令行参数
input_file = sys.argv[1]
output_dir = sys.argv[2]

# 创建测试数据集和数据加载器
test_dataset = TestDataset(input_file)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# 进行预测并保存结果
print("开始预测")
predict(model, test_dataloader, output_dir)
print(f"预测结果已保存到 {output_dir}")