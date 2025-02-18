import os
import cv2
import json

def split_video(input_video_path, output_dir, frames_per_chunk=1000):
    """
    주어진 비디오 파일을 frames_per_chunk 단위로 분할합니다.
    반환값은 성공 여부, 오류 코드, 메시지, 그리고 생성된 chunk 파일들의 경로를 담은 JSON 객체입니다.
    """
    cap = cv2.VideoCapture(input_video_path)
    
    if not cap.isOpened():
        error_message = f"Error: 비디오를 열 수 없습니다. ({input_video_path})"
        print(error_message)
        return {
            "status": "error",
            "error_code": 1,
            "message": error_message,
            "chunks": []
        }

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_name = os.path.basename(input_video_path).split('.')[0]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    frame_count = 0
    chunk_index = 0
    frames = []
    chunk_files = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frames.append(frame)
        frame_count += 1

        if frame_count % frames_per_chunk == 0 or frame_count == total_frames:
            chunk_index += 1
            chunk_output_path = os.path.join(output_dir, f"{video_name}_chunk{chunk_index}.mp4")
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(chunk_output_path, fourcc, fps, (frame.shape[1], frame.shape[0]))

            for f in frames:
                out.write(f)
            out.release()

            print(f"'{chunk_output_path}' 저장 완료.")
            chunk_files.append(chunk_output_path)
            frames = []

    cap.release()
    message = f"{input_video_path}: 비디오 분할 완료. 총 {chunk_index}개의 chunk 생성됨."
    print(message)
    return {
        "status": "success",
        "error_code": 0,
        "message": message,
        "chunks": chunk_files
    }

def split_all_videos_in_directory(video_dir, output_dir, frames_per_chunk=1000):
    """
    지정된 디렉토리 내의 모든 비디오 파일을 분할하며, 각 파일에 대해 split_video 함수를 호출합니다.
    반환값은 각 비디오 파일에 대한 처리 결과를 담은 JSON 객체입니다.
    """
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    video_files = [f for f in os.listdir(video_dir) if f.lower().endswith(video_extensions)]
    
    if not video_files:
        message = "⚠️ 처리할 비디오 파일이 없습니다."
        print(message)
        return {
            "status": "error",
            "error_code": 2,
            "message": message,
            "results": {}
        }

    print(f"🎥 총 {len(video_files)}개의 비디오를 처리합니다.\n")
    results = {}
    for video in video_files:
        video_path = os.path.join(video_dir, video)
        print(f"▶ '{video}' 처리 시작...")
        result = split_video(video_path, output_dir, frames_per_chunk)
        results[video] = result
    
    final_message = "모든 비디오 처리 완료."
    print(final_message)
    return {
        "status": "success",
        "error_code": 0,
        "message": final_message,
        "results": results
    }

# 사용 예시:
if __name__ == "__main__":
    # 단일 비디오 파일에 대해 분할 수행 (입력 경로와 출력 디렉토리 지정)
    input_video_path = "./video/example.mp4"  # 분할할 비디오 파일 경로
    output_dir = "./video_split"              # 분할된 비디오를 저장할 디렉토리
    result = split_video(input_video_path, output_dir, frames_per_chunk=500)
    print(json.dumps(result, ensure_ascii=False, indent=4))

    # 또는 디렉토리 내의 모든 비디오 파일을 처리하려면:
    # video_dir = './video'
    # output_dir = './video_split'
    # result = split_all_videos_in_directory(video_dir, output_dir, frames_per_chunk=500)
    # print(json.dumps(result, ensure_ascii=False, indent=4))
