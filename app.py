from flask import Flask, render_template, request, redirect, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import torch
import torch.nn as nn
import pickle
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ================= APP =================
app = Flask(__name__)
app.secret_key = "secret123"

# ================= DATABASE =================
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)

# ================= LOGIN =================
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# ================= USER MODEL =================
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100))
    password = db.Column(db.String(100))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ================= LOAD ML FILES =================
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
encoder = pickle.load(open("label_encoder.pkl", "rb"))

vocab_size = 10000
max_len = 60
num_classes = len(encoder.classes_)

# ================= MODEL =================
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(10000, 128)
        self.lstm = nn.LSTM(128, 128, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = LSTMModel()
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
model.eval()

# ================= CLEAN TEXT =================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ================= EMOJI MAP =================
emoji_map = {
    "happy": "😊",
    "sad": "😢", 
    "angry": "😡",
    "fear": "😨",
    "love": "❤️",
    "surprise": "😲",
    "neutral": "😐"
}

# ================= RECOMMENDATIONS =================
recommendations = {
    "happy": "💡 Keep smiling and spread positivity!",
    "sad": "💡 Talk to someone you trust ❤️",
    "angry": "💡 Take deep breaths and relax 😌",
    "fear": "💡 Stay calm and try meditation 🧘",
    "love": "💡 Share your happiness with others ❤️",
    "surprise": "💡 Take a moment to process your feelings 😲",
    "neutral": "💡 Stay balanced and positive 👍"
}

# ================= ROUTES =================

# ✅ ALWAYS START WITH LOGIN PAGE
@app.route("/")
def home():
    return redirect("/login")

# -------- LOGIN --------
@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect("/dashboard")   # already logged in

    if request.method == "POST":
        user = User.query.filter_by(
            username=request.form["username"],
            password=request.form["password"]
        ).first()

        if user:
            login_user(user)
            return redirect("/dashboard")

        return "Invalid login ❌"

    return render_template("login.html")

# -------- REGISTER --------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        new_user = User(
            username=request.form["username"],
            password=request.form["password"]
        )
        db.session.add(new_user)
        db.session.commit()
        return redirect("/login")

    return render_template("register.html")

# -------- DASHBOARD (MAIN PAGE) --------
@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("index.html")

# -------- LOGOUT --------
@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect("/login")

# -------- PREDICT --------
@app.route("/predict", methods=["POST"])
@login_required
def predict():
    text = clean_text(request.form["text"])

    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len)

    with torch.no_grad():
        input_tensor = torch.tensor(padded, dtype=torch.long)
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probs, 1)

    emotion = encoder.inverse_transform([pred.item()])[0]
    print("Predicted emotion:", emotion)

    emoji = emoji_map.get(emotion.lower(), "🙂")
    recommendation = recommendations.get(emotion.lower(), "")

    return jsonify({
        "prediction": emotion,
        "emoji": emoji,
        "confidence": round(confidence.item() * 100, 2),
        "recommendation": recommendation
    })

# ================= RUN =================
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)