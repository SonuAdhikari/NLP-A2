import pickle
from flask import Flask, render_template, request
from utils import LSTMLanguageModel, generate
import torch 

app = Flask(__name__)

# Load the data during initialization
with open('lstm.pkl', 'rb') as data_file:
    lstm = pickle.load(data_file)

# Extracting data for the language model
vocab_size = lstm['vocab_size']
emb_dim = lstm['emb_dim']
hid_dim = lstm['hid_dim']
num_layers = lstm['num_layers']
dropout_rate = lstm['dropout_rate']
tokenizer = lstm['tokenizer']
vocab = lstm['vocab']

# Instantiate the model
model = LSTMLanguageModel(vocab_size, emb_dim, hid_dim, num_layers, dropout_rate)
# Assuming you have a saved model, load the state dict here
model.load_state_dict(torch.load('best-val-harry_potter_lstm_lm.pt', map_location=torch.device('cpu')))
model.eval()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    elif request.method == 'POST':
        prompt = request.form['prompt']
        seq_len = 40
        temperature = 0.7
        seed = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Instantiate the model and load the state dict if needed
        generation = generate(prompt, seq_len, temperature, model, tokenizer, 
                              vocab, device, seed)
        
        sentence = ' '.join(generation)
        return render_template('index.html', prompt=prompt, seq_len=seq_len, sentence=sentence)

if __name__ == "__main__":
    app.run(debug=True)
