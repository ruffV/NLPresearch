from youtube_transcript_api import YouTubeTranscriptApi

tx = YouTubeTranscriptApi.get_transcript("Aieb4EjVKNQ")#skete smd video https://www.youtube.com/watch?v=lBV8SS9zhWc

for i in tx:
	print(i['text']) # produces wall of text of pete davidson stand up show


#ntlk uses tuples like this:  [("how did the chikcne uhh???",  "Banana!"), ("another one)]