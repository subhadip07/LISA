import os,sys
from pathlib import Path
import logging

# variables to describe all the files and folders that will be used in the project
while True:
    project_name=input("Enter your project name: ")
    if project_name!="":
        break

list_of_files=[
    f"{project_name}/__init__.py",
    f"{project_name}/components/__init__.py",
    f"{project_name}/config/__init__.py",
    f"{project_name}/constants/__init__.py",
    f"{project_name}/entity/__init__.py",
    f"{project_name}/exception/__init__.py",
    f"{project_name}/logger/__init__.py",
    f"{project_name}/pipeline/__init__.py",
    f"{project_name}/utils/__init__.py",
    f"config/config.yml",
    "schema.yml",
    "app.py",
    "main.py",
    "logs.py",
    "exception.py",
    "setup.py"
]

for file_path in list_of_files:
    filepath=Path(file_path)
    filedir,filename=os.path.split(filepath)
    
    if filedir !="":
        os.makedirs(filedir,exist_ok=True)
    
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):
        with open(filepath,"w") as f:
            pass
        
    else:
        logging.info(f"File is already available at :{filepath}") 