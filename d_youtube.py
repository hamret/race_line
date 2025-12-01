import yt_dlp

def download_youtube_video(url: str, output_path: str = "./videos", filename: str = "f1_onboard.mp4"):
    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",  # 가장 좋은 mp4 품질
        "outtmpl": f"{output_path}/{filename}",  # 저장 경로 + 파일명
        "merge_output_format": "mp4",  # 비디오/오디오 합쳐서 mp4로
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

if __name__ == "__main__":
    url = "https://youtu.be/V0-xztTNyZc?si=KCgPq5-05vkockk0"  # 다올님 F2 오버테이크 영상
    download_youtube_video(url)
    print("다운로드 완료!")
