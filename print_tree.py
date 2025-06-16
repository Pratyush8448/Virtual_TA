import os

def print_tree(startpath, max_depth=2, indent="  ", depth=0):
    for item in sorted(os.listdir(startpath)):
        if item in {".git", ".venv", "__pycache__"}:
            continue
        path = os.path.join(startpath, item)
        if os.path.isdir(path):
            print(f"{indent * depth}📁 {item}/")
            if depth + 1 < max_depth:
                print_tree(path, max_depth, indent, depth + 1)
        else:
            print(f"{indent * depth}📄 {item}")

print_tree(".", max_depth=2)
