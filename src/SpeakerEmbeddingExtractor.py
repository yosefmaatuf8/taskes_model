import os
import pandas as pd
from pydub import AudioSegment
from embedding_handler import EmbeddingHandler  # ודא שהמחלקה EmbeddingHandler נמצאת בקובץ הנכון
from globals import GLOBALS
from db_manager.db_manager import DBManager


class SpeakerEmbeddingExtractor:
    def __init__(self, audio_folder, output_csv, temp_wav_folder="temp_wav"):
        """
        אתחול המחלקה, עם תיקייה זמנית להמרת קבצים ל-WAV.
        """
        self.audio_folder = audio_folder
        self.db_manager = DBManager()
        self.output_csv = output_csv
        self.temp_wav_folder = temp_wav_folder
        self.embedding_handler = EmbeddingHandler(self.db_manager)  # שימוש במחלקה להפקת אמבדינגים

        # יצירת תיקייה זמנית להמרת קבצים
        os.makedirs(self.temp_wav_folder, exist_ok=True)

    def convert_to_wav(self, file_path):
        """
        ממיר קובץ אודיו ל-WAV.
        """
        audio = AudioSegment.from_file(file_path)
        wav_path = os.path.join(self.temp_wav_folder, os.path.basename(file_path).replace(".mp3", ".wav"))
        audio.export(wav_path, format="wav")
        return wav_path

    def extract_embeddings(self):
        """
        קורא את כל קבצי האודיו מהתיקייה, ממיר ל-WAV אם צריך, מחלץ את האמבדינג של כל דובר ושומר את הנתונים לקובץ CSV.
        """
        data = []

        for file_name in os.listdir(self.audio_folder):
            if file_name.endswith(".mp3") or file_name.endswith(".wav"):
                speaker_name = file_name.split("_")[0]  # שם הדובר מתוך שם הקובץ
                file_path = os.path.join(self.audio_folder, file_name)

                # המרה ל-WAV אם הקובץ בפורמט MP3
                if file_name.endswith(".mp3"):
                    print(f"🔄 Converting {file_name} to WAV...")
                    file_path = self.convert_to_wav(file_path)

                # טוען את קובץ האודיו
                audio_clip = AudioSegment.from_wav(file_path)

                # מחלץ את האמבדינג
                embedding = self.embedding_handler.extract_embedding(audio_clip)

                if embedding is not None:
                    data.append({"speaker": speaker_name, "embedding": embedding.tolist()})

        # שמירת הנתונים לקובץ CSV
        df = pd.DataFrame(data)
        df.to_csv(self.output_csv, index=False)

        print(f"✅ Embeddings saved to {self.output_csv}")


# דוגמה לשימוש
if __name__ == "__main__":
    audio_folder = "/home/mefathim/PycharmProjects/taskes_model/db/audio_for_speaker"  # שנה לתיקייה שבה קבצי האודיו שלך
    output_csv = "speaker_embeddings.csv"

    extractor = SpeakerEmbeddingExtractor(audio_folder, output_csv)
    extractor.extract_embeddings()
