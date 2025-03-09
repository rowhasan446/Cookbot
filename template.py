import os
from pathlib import Path
#load env
from dotenv import load_dotenv
load_dotenv()

#logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

list_of_files = [
  'src/main.py',
  'src/__init__.py',
  'src/helper.py',
  ".env",
  "requirements.txt",
  "Dockerfile",
  "docker-compose.yml",
  "setup.py",
  "README.md",
  "app.py",
  "research/trials.ipynb",
]

for file in list_of_files:
  filepaths =Path(file)
  filedir, filename = os.path.split(filepaths)

  if filedir !="":
    os.makedirs(filedir, exist_ok=True)
    logging.info(f"Creating directory {filedir}")
  
  if(not os.path.exists(filepaths) or (os.path.getsize(filepaths) == 0)):
    logging.info(f"Creating file {filename}")
    with open(filepaths, "w") as f:
     pass
     logging.info(f"Created file {filename}")
  else:
    logging.info(f"File {filename} already exists")
