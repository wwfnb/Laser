export train_yaml="config/train_full/multi_dpo.yaml"
echo "train_yaml: ${train_yaml}"
FORCE_TORCHRUN=1 llamafactory-cli train $train_yaml
