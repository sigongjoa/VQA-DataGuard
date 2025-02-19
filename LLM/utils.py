import json
import datetime

def save_config_to_file(config, base_filename="chain_results"):
    """
    구성(config) 정보를 JSON 파일로 저장합니다.
    저장 파일명은 base_filename에 현재 시간이 추가된 형태로 생성됩니다.
    
    Args:
        config (dict): 저장할 구성 정보.
        base_filename (str): 파일명 기본 이름 (기본값 "chain_results").
        
    Returns:
        str: 저장된 파일명.
    """
    # 현재 시간을 "YYYYMMDD_HHMMSS" 형식으로 포맷팅
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_filename}_{current_time}.json"
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    
    print(f"Results saved to {filename}")
    return filename