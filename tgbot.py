# import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler
from telegram.ext import filters, MessageHandler

import sys
import os
from pursuit import get_closest_to_file
from tempfile import NamedTemporaryFile

# logging.basicConfig(
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     level=logging.INFO
# )


async def photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    new_file = await update.message.effective_attachment.get_file()
    # update.message.text
    try:
        with NamedTemporaryFile(delete=True) as f:
            bs = await new_file.download_as_bytearray()
            f.write(bs)
            rows = get_closest_to_file(f.name)
    except Exception as e:
        print(e, file=sys.stderr)
        await context.bot.send_message(
            chat_id=update.effective_chat.id, text=f"Error processing image: {e}"
        )
        return
    characters = {r["char"] for r in rows}
    msg = f"Closest characters: {characters}"
    await context.bot.send_message(chat_id=update.effective_chat.id, text=msg)


async def help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="I'm a bot to detect fursuiters, please send a photo to me!",
    )


if __name__ == "__main__":
    token = os.environ.get("TG_TOKEN")
    application = ApplicationBuilder().token(token).build()

    photo_handler = MessageHandler((~filters.COMMAND) & filters.PHOTO, photo)
    application.add_handler(photo_handler)

    help_handler = CommandHandler("help", help)
    application.add_handler(help_handler)

    application.run_polling()
