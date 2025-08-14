import pandas as pd
import string
import nltk
import pyttsx3
import speech_recognition as sr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download stopwords
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))

#Read CSV
faq_df = pd.read_csv("faq_dataset.csv")
faq_df.columns = faq_df.columns.str.lower().str.strip()

if 'question' not in faq_df.columns or 'answer' not in faq_df.columns:
    raise ValueError("CSV must have 'question' and 'answer' columns")

#Text Preprocessing 
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)

faq_df['clean_question'] = faq_df['question'].apply(preprocess)

# TF-IDF Vectorization 
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(faq_df['clean_question'])

# Text-to-Speech 
engine = pyttsx3.init()
engine.setProperty('rate', 180)
engine.setProperty('volume', 1.0)

def speak(text):
    engine.say(text)
    engine.runAndWait()

#  Get Best Match 
def get_answer(user_q):
    user_clean = preprocess(user_q)
    user_vec = vectorizer.transform([user_clean])
    similarity = cosine_similarity(user_vec, X).flatten()
    max_score_index = similarity.argmax()
    if similarity[max_score_index] > 0.3:  # Threshold for match
        return faq_df.iloc[max_score_index]['answer']
    else:
        return "Sorry, I couldn't find a relevant answer."

# Speech Input 
def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("\n🎤 Listening... Speak now!")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
    try:
        text = r.recognize_google(audio)
        print(f"Your Question is: {text}")
        return text
    except:
        print("❌ Could not understand your voice.")
        return ""

# Main Chat Loop 
if __name__ == "__main__":
    print("🤖 Welcome to AI & ML Jobs FAQ Chatbot! (Type 'exit' to quit)\n")
    mode = input("Do you want to ask by 'speech' or 'text'? ").strip().lower()

    while True:
        if mode == "speech":
            user_q = listen()
        else:
            user_q = input("\nYou: ")

        if user_q.lower() == "exit":
            print("Chatbot: Goodbye!")
            speak("Goodbye!")
            break
        elif user_q.strip() == "":
            continue

        answer = get_answer(user_q)
        print(f"Chatbot: {answer}")
        speak(answer)
