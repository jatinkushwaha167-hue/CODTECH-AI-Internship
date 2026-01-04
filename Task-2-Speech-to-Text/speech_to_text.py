import speech_recognition as sr

r = sr.Recognizer()

with sr.AudioFile("audio.wav") as source:
    print("Listening audio...")
    audio = r.record(source)

try:
    text = r.recognize_google(audio)
    print("Converted Text:")
    print(text)

except sr.UnknownValueError:
    print("Audio samajh nahi aaya")

except sr.RequestError:
    print("Internet / API issue")
