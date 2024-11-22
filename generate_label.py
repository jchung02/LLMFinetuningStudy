import os
import json

refer_path = [
    "/Users/hayoun/Documents/vscode/24-2-project/finetuning-study/158.시간 표현 탐지 데이터/01-1.정식개방데이터/Training/02.라벨링데이터",
    "/Users/hayoun/Documents/vscode/24-2-project/finetuning-study/158.시간 표현 탐지 데이터/01-1.정식개방데이터/Validation/02.라벨링데이터"
]

# 라벨 저장 세트
labels_set = set()

def navigate_and_extract_labels(base_path, cur_relative_path, labels_set):
    cur_path = os.path.join(base_path, cur_relative_path)
    sub_dirs = os.listdir(cur_path)
    sub_dirs = [x for x in sub_dirs if os.path.isdir(os.path.join(cur_path, x))]

    # 최종 디렉토리 도달: JSON 파일에서 라벨 추출
    if not sub_dirs:
        json_files = [f for f in os.listdir(cur_path) if f.endswith('.json')]
        for json_file in json_files:
            json_path = os.path.join(cur_path, json_file)
            with open(json_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    for annotation in data.get("annotations", []):
                        labels_set.add(annotation["label"])
                except Exception as e:
                    print(f"Error reading {json_path}: {e}")
        return labels_set

    # 하위 디렉토리 순회
    for d in sub_dirs:
        labels_set = navigate_and_extract_labels(
            base_path,
            os.path.join(cur_relative_path, d),
            labels_set
        )
    return labels_set

if __name__ == '__main__':
    for p in refer_path:
        labels_set = navigate_and_extract_labels(p, '', labels_set)

    # 라벨을 파일로 저장
    label_file_path = "/Users/hayoun/Documents/vscode/24-2-project/finetuning-study/158.시간 표현 탐지 데이터/labels.txt"
    with open(label_file_path, 'w', encoding='utf-8') as f:
        for label in sorted(labels_set):
            f.write(label + "\n")
    print(f"Labels saved to {label_file_path}")

