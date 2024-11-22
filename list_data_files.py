import os
import pdb

refer_path = [
        "/Users/hayoun/Documents/vscode/24-2-project/finetuning-study/158.시간 표현 탐지 데이터/01-1.정식개방데이터/Training/02.라벨링데이터",
        # "./data_path/Test/라벨링데이터/",
        "/Users/hayoun/Documents/vscode/24-2-project/finetuning-study/158.시간 표현 탐지 데이터/01-1.정식개방데이터/Validation/02.라벨링데이터"
]
docs_names = set()

def navigate_directory(base_path, cur_relative_path, path_collection):
    cur_path = base_path + '/' + cur_relative_path
    sub_dirs = os.listdir(cur_path)
    sub_dirs = [x for x in sub_dirs if os.path.isdir(cur_path + '/' + x)]
    if not sub_dirs:
        return path_collection + [cur_relative_path]
    for d in sub_dirs:
        path_collection = navigate_directory(
            base_path,
            cur_relative_path + '/' + d,
            path_collection)
    return path_collection

if __name__ == '__main__':
    all_paths = []
    for p in refer_path:
        tmp_paths = navigate_directory(p, '', [])
        all_paths += tmp_paths
    pdb.set_trace()
