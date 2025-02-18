import os
import json

def create_or_read_json(video_name, json_dir="./json_files"):
    """
    주어진 video_name에 대응하는 JSON 파일을 확인합니다.
    - 파일이 없으면, 아래와 같은 포맷으로 모든 success 값을 False로 설정하여 생성합니다.
    - 파일이 존재하면 파일을 읽어 반환합니다.
    
    JSON 포맷:
    {
        "split_video": {"success": False, "output_path": ""},
        "alphapose": {"success": False, "output_path": ""},
        "motionBERT": {"success": False, "output_path": ""},
        "PoseC3D": {"success": False, "output_path": ""}
    }
    """
    # JSON 파일을 저장할 디렉토리 생성
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)
    
    json_file = os.path.join(json_dir, f"{video_name}.json")
    
    if os.path.exists(json_file):
        # JSON 파일이 있으면 읽어서 반환
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"JSON 파일 '{json_file}'이(가) 존재하여 내용을 읽어들였습니다.")
    else:
        # JSON 파일이 없으면 지정한 포맷으로 생성
        data = {
            "split_video": {"success": False, "output_path": ""},
            "alphapose": {"success": False, "output_path": ""},
            "motionBERT": {"success": False, "output_path": ""},
            "PoseC3D": {"success": False, "output_path": ""}
        }
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"JSON 파일 '{json_file}'이(가) 없어 새로 생성하였습니다.")
    
    return data