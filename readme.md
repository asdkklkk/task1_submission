### 该程序使用于CLEF2025-task1

tira-cli code-submission \
	--path . \
	--task generative-ai-authorship-verification-panclef-2025 \
	--dataset pan25-generative-ai-detection-smoke-test-20250428-training \
	--mount-hf-model kkkkl5/asdkklkk \
	--command 'python3 task1.2_int.py $inputDataset/dataset.json $outputDir/predictions.jsonl' --dry-run
