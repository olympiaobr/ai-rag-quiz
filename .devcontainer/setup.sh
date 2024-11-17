python -m venv .venv

source .venv/bin/activate
echo "source .venv/bin/activate" >> $HOME/.bashrc

pip install --upgrade pip
pip install -r requirements.txt
pip install -e . 
