"""
The script generates the 
repository structure 
"""

import os


dirs = [                                    # directories to be created
    os.path.join("data", "raw"),
    os.path.join("data", "processed"),
    "notebooks",
    "saved_models",
    "src"    
]

for dir_ in dirs:                       # the loop prevents existing folders to change on script iteration
    os.makedirs(dir_, exist_ok=True)   
    with open(os.path.join(dir_, "gitkeep"  ), "w") as f:
        pass


files = [
    "dvc.yaml",
    "params.yaml",
    ".gitignore",
    os.path.join("src","__init__.py"),                # for source to be read as python package
]

for file_ in files:
    with open(file_, "w") as f:
        pass