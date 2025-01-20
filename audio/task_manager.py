from odf.opendocument import OpenDocumentText, load
from odf.text import P

class TaskManager:
    def save_to_odt(self, text, filename):
        doc = OpenDocumentText()
        for line in text.split("\n"):
            if line.strip():
                doc.text.addElement(P(text=line))
                doc.text.addElement(P(text=""))
        doc.save(filename)

    def extract_text_from_odt(self, filename):
        doc = load(filename)
        return "\n".join(str(p) for p in doc.getElementsByType(P))

    def extract_tasks(self, summarized_text):
        # Example logic for extracting tasks
        return {"completed": ["Task 1", "Task 2"], "to_do": ["Task 3", "Task 4"]}
