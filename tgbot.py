# import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler
from telegram.ext import filters, MessageHandler

import sys
import os
from pursuit import detect_characters
from tempfile import NamedTemporaryFile

# logging.basicConfig(
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     level=logging.INFO
# )


async def photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not isinstance(update.message.effective_attachment, list):
        print("Invalid attachment type", file=sys.stderr)
        return
    attachment = update.message.effective_attachment
    new_file = await attachment[-1].get_file()  # get largest file
    # update.message.text
    try:
        with NamedTemporaryFile(delete=True) as f:
            bs = await new_file.download_as_bytearray()
            f.write(bs)
            rows = detect_characters(f.name, 5)
    except Exception as e:
        print(e, file=sys.stderr)
        await context.bot.send_message(
            chat_id=update.effective_chat.id, text=f"Error processing image: {e}"
        )
        return
    characters = {r["char"] for r in rows}
    msg = f"Closest characters: {characters}"
    await context.bot.send_message(chat_id=update.effective_chat.id, text=msg)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="I'm a bot that identifies fursuiters in pictures, please send a photo to me!",
    )


if __name__ == "__main__":
    token = os.environ.get("TG_BOT_TOKEN", "")
    application = ApplicationBuilder().token(token).build()

    photo_handler = MessageHandler((~filters.COMMAND) & filters.PHOTO, photo)
    application.add_handler(photo_handler)

    start_handler = CommandHandler("start", start)
    application.add_handler(start_handler)

    application.run_polling()
