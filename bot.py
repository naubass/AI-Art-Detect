import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
import torch
from torchvision import models, transforms
from PIL import Image
import io
import os
from dotenv import load_dotenv
load_dotenv()

# config
TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
MODEL_PATH = 'model_art_ai_human.pth'

# Setup Logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Load model AI
print("Loading model AI...")
device = torch.device('cpu')

try:
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    class_names = checkpoint['class_names']

    # setup arsitektur model
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, len(class_names))

    # Load weights
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print(f"Model berhasil di-load! Kelas: {class_names}")

except FileNotFoundError:
    print(f"Model {MODEL_PATH} tidak ditemukan!")
    exit(1)

# Transformasi gambar
img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Fungsi untuk melakukan prediksi
def predict_image(image_bytes):
    try:
        # Membaca gambar
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Preprocess Image
        tensor = img_transform(image).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            score, preds = torch.max(probabilities, 1)

        predicted_class = class_names[preds[0]]
        confidence = score.item() * 100

        return predicted_class, confidence
    except Exception as e:
        print(f"Error Prediction: {e}")
        return None, 0
    
# Telegram Handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user.first_name
    await update.message.reply_text(
        f"Halo {user}! 👋\n\n"
        "Saya adalah Bot Deteksi Gambar AI vs Human Art.\n"
        "Kirimkan foto buah kepada saya, dan saya akan menebak kondisinya (Segar/Rusak)."
    )

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    status_msg = await update.message.reply_text("🔍 Sedang mengamati gambar...")
    photo_file = await update.message.photo[-1].get_file()
    image_bytes = await photo_file.download_as_bytearray()

    # prediction
    label, conf = predict_image(image_bytes)

    if label:
        # Logika baru
        if "ai" in label.lower():
            status = "🤖 AI GENERATED"
            desc = "Terdeteksi pola artifisial (AI)."
            warning = "⚠️ Kemungkinan besar gambar ini dibuat oleh komputer."
        else:
            status = "🎨 HUMAN ART"
            desc = "Terdeteksi goresan tangan manusia."
            warning = "✅ Kemungkinan besar ini karya ilustrator asli."

        result_text = (
            f"🔍 **Analisa Selesai**\n"
            f"Hasil: **{status}**\n"
            f"Confidence: {conf:.1f}%\n\n"
            f"{desc}\n{warning}"
        )    
    else:
        result_text = "🔍 Analisa Selesai. Tidak dapat menentukan jenis gambar."

    await status_msg.edit_text(result_text, parse_mode="Markdown")

# Main Function
if __name__ == "__main__":
    app = ApplicationBuilder().token(TOKEN).build()
    
    app.add_handler(CommandHandler('start', start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    
    print("Bot Telegram berjalan... Tekan Ctrl+C untuk stop.")
    app.run_polling()