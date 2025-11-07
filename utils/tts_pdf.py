import io
from gtts import gTTS
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

def text_to_speech_bytes(text: str) -> bytes:
    tts = gTTS(text=text, lang='en', slow=False)
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    return fp.read()

def text_to_pdf_bytes(text: str) -> bytes:
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    text_obj = p.beginText(50, height - 50)
    text_obj.setFont("Helvetica", 11)
    text_obj.setLeading(14)
    
    for line in text.splitlines():
        if line.strip():
            text_obj.textLine(line[:100])  # Prevent overflow
        else:
            text_obj.textLine("")
    
    p.drawText(text_obj)
    p.showPage()
    p.save()
    buffer.seek(0)
    return buffer.read()