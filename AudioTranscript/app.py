from moviepy.editor import VideoFileClip
import speech_recognition as sr

def extract_audio(video_path, audio_path):
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(audio_path)

def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    audio_file = sr.AudioFile(audio_path)

    with audio_file as source:
        audio = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio, language='fr-FR')
        return text
    except sr.UnknownValueError:
        return "Google Speech Recognition could not understand the audio"
    except sr.RequestError as e:
        return f"Could not request results from Google Speech Recognition service; {e}"

def save_transcription(text, file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text)

# Main function
def main(video_path, transcription_path):
    audio_path = "output_audio.wav"
    extract_audio(video_path, audio_path)
    transcription = transcribe_audio(audio_path)
    save_transcription(transcription, transcription_path)

# Example usage
main("path_to_video.mp4", "transcription.txt")
