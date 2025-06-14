import json
import sys

def fix_notebook_widgets(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Remove problematic widget metadata while preserving outputs
    if 'metadata' in nb and 'widgets' in nb['metadata']:
        # Clear widget state but keep outputs
        nb['metadata']['widgets'] = {}
    
    # Check each cell for widget metadata issues
    for cell in nb.get('cells', []):
        if 'metadata' in cell and 'widgets' in cell['metadata']:
            cell['metadata']['widgets'] = {}
    
    # Save the fixed notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)

if __name__ == "__main__":
    fix_notebook_widgets("evaluation/explainability.ipynb")
    print("Fixed widget metadata issues")