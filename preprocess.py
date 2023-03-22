import pytube
from pytube import YouTube

def preprocessor(url):
  vid_url=YouTube(url)
  stream=vid_url.streams.filter(only_audio=True)
  stream=stream.first()
  download=stream.download(filename="yt.mp4")

  return download