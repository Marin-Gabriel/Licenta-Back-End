from googletrans import Translator
translator = Translator()

translatedText=translator.translate("yo soy cansado",dest='ro')

print(translatedText.text)