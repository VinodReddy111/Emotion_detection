import re
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer

# ---------- Data Reading ----------
def read_data(file):
    data = []
    with open(file, 'r') as f:
        for line in f:
            line = line.strip()
            label = ' '.join(line[1:line.find("]")].strip().split())
            text = line[line.find("]")+1:].strip()
            data.append([label, text])
    return data

# ---------- Preprocessing ----------
def ngram(token, n):
    return [' '.join(token[i-n+1:i+1]) for i in range(n-1, len(token))]

def create_feature(text, nrange=(1, 1)):
    text = text.lower()
    text_alphanum = re.sub('[^a-z0-9#]', ' ', text)
    text_features = []
    for n in range(nrange[0], nrange[1]+1):
        text_features += ngram(text_alphanum.split(), n)
    text_punc = re.sub('[a-z0-9]', ' ', text)
    text_features += ngram(text_punc.split(), 1)
    return Counter(text_features)

def convert_label(item, name):
    items = list(map(float, item.split()))
    return ' '.join([name[i] for i in range(len(items)) if items[i] == 1])

# ---------- Load & Process Data ----------
emotions = ["joy", "fear", "anger", "sadness", "disgust", "shame", "guilt"]
file_path = "C:\Users\vinod\Downloads\text.t1.zip"  # 🔁 Update with your actual dataset path
data = read_data(file_path)

X_all = []
y_all = []
for label, text in data:
    y_all.append(convert_label(label, emotions))
    X_all.append(create_feature(text, nrange=(1, 4)))

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.3, random_state=0)

# ---------- Vectorization ----------
vectorizer = DictVectorizer(sparse=True)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# ---------- Models ----------
models = [
    SVC(),
    LinearSVC(random_state=0),
    RandomForestClassifier(random_state=0),
    DecisionTreeClassifier()
]

# ---------- Training & Evaluation ----------
print("| {:25} | {:17} | {:13} |".format("Classifier", "Training Accuracy", "Test Accuracy"))
print("| {} | {} | {} |".format("-"*25, "-"*17, "-"*13))
for clf in models:
    clf.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))
    print("| {:25} | {:17.7f} | {:13.7f} |".format(clf.__class__.__name__, train_acc, test_acc))

# ---------- Dynamic Input ----------
emoji_dict = {
    "joy": "😂", "fear": "😱", "anger": "😠", "sadness": "😢",
    "disgust": "😒", "shame": "😳", "guilt": "😔"
}
chosen_clf = models[1]  # Using LinearSVC

print("\nEmotion Prediction - Type a sentence (type 'exit' to stop):")
while True:
    user_text = input("\nEnter text: ")
    if user_text.lower() == "exit":
        break
    features = vectorizer.transform([create_feature(user_text, nrange=(1, 4))])
    prediction = chosen_clf.predict(features)[0]
    print(f"Predicted Emotion: {prediction} {emoji_dict.get(prediction, '')}")
