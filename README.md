# Inverse folding with simple GNN

### Установка окружения

```bash
conda env create -f environment.yaml
conda activate gnn_workshop
```
Обновление окружения (если поменяли какие-то зависимости):

```bash
conda env update -n gnn_workshop -f environment.yaml
```

### Запуск

Визуализация графа:
```bash
python scripts/visualize.py
```


Обучение:
```bash
python scripts/train.py
```

Логи обучения:
```bash
aim up --repo logs/.aim
```
