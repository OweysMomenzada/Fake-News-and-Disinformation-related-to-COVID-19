# Fake Covid App
This file contains an installation guid to setup and run the fakecovapp.

## 1. Install
1. Setup new environment via python venv or anaconda:
python
```
python3 -m venv myvenv
```
anaconda
```shell
conda create --name myenv
```
2. Activate your venv:
python venv
```shell
source /path/to/venv/bin/activate
```
or via anaconda
```
conda activate /path/to/condavenv/
```
3. Install requirements:
````shell
pip install -r /path/to/requirements.txt
````
## 2. Run
1. Activate venv (see 1.2)
2. Move to directory
```shell
cd /your/dir/FakeNewsDetectionSystem/fakecovapp
```
3. Run app
```shell
python app.py
```