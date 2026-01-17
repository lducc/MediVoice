import json
from typing import Dict, Any
from openai import OpenAI
import configs
from llm_confidence.logprobs_handler import LogprobsHandler 

client = OpenAI(api_key = configs.OPENAI_KEY)
logprobs_handler = LogprobsHandler()

def format_conversation(segments: list) -> str:
    #Format diarized segments into a conversation
    if not segments:
        return ""
    
    conversation = []
    current_speaker = None
    current_text = []
    
    for seg in segments:
        speaker = seg.get('speaker', 'UNKNOWN')
        text = seg.get('text', '').strip()
        
        if not text:
            continue
        
        if speaker == current_speaker:
            current_text.append(text)
        else:
            if current_speaker and current_text:
                conversation.append(f"{current_speaker}: {' '.join(current_text)}")
            
            current_speaker = speaker
            current_text = [text]
    
    # Add final speaker
    if current_speaker and current_text:
        conversation.append(f"{current_speaker}: {' '.join(current_text)}")
    
    return "\n\n".join(conversation)


def extract_medical_data(transcript: str = None, segments: list = None, lang: str = "vi") -> Dict[str, Any]:    
    system_prompt = """
    Bạn là một nhân viên y khoa AI chuyên nghiệp tên là MediVoice.
    Nhiệm vụ của bạn là phân tích cuộc hội thoại khám bệnh và trích xuất thông tin y tế có cấu trúc.

    Định dạng đầu ra: Kết quả trả về bắt buộc phải là định dạng JSON hợp lệ (không có markdown block), tuân theo cấu trúc chính xác sau:
    {
      "patient_info": { 
          "age": int or null, 
          "gender": "Nam"/"Nữ" or null, 
          "nationality": str or null 
      },
      "chief_complaint": str (Trích dẫn lý do đi khám mà bệnh nhân một cách cụ thể và đầy đủ),
      "hpi": { 
        "duration": str or null (VD: "5 ngày nay", "3 tuần"), 
        "symptoms": List[str] (Các triệu chứng cụ thể: "khó thở", "ho", "sốt"), 
        "negative_symptoms": List[str] (Triệu chứng bệnh nhân phủ nhận: "không sốt", "không đau họng"), 
        "description": str (Trích dẫn chính xác đầy đủ lời kể của bệnh nhân về diễn biến bệnh) 
      },
      "past_medical_history": { 
          "chronic_diseases": List[str] (Tên bệnh mãn tính mà bệnh nhân đã từng có), 
          "allergies": List[str] (Dị ứng cụ thể), 
          "current_medications": List[str] (Thuốc đang dùng, kèm liều lượng nếu có) 
      },
      "assessment": List[str] (Chẩn đoán của bác sĩ, trích dẫn chính xác tên bệnh từ transcript),
      "plan": { 
          "tests": List[str] (Các xét nghiệm cụ thể, VD: ["xét nghiệm máu", "X-quang phổi"]), 
          "medications": List[str] (Tên thuốc có thật + liều lượng ghi chính xác và đầy đủ), 
          "advice": List[str] (Advice của bác sĩ tách riêng thành nhiều mục khác nhau, mỗi mục từ 2-6 từ và cụ thể. VD: ["Ăn nhạt", "Kê cao gối", "Tái khám 1 tuần"]) 
      },
      "missing_fields": List[str] (Liệt kê các thông tin (key) thiếu/ null, ví dụ: "age", "allergies", "nationality")
    }
    
    Ghi chú:
        Đối với các trường mô tả, hãy trích dẫn chính xác từ hội thoại đầy đủ.
        Nếu có nhiều người nói và có nhãn SPEAKER_XX, hãy sử dụng ngữ cảnh để xác định ai là bác sĩ, ai là bệnh nhân.
        Nếu thông tin không có, để null hoặc danh sách rỗng.
    """

    try:
        if segments and len(segments) > 0:
            conversation = format_conversation(segments)
            input_text = conversation
            source_type = "conversation"

        elif transcript:
            input_text = transcript
            source_type = "transcript"
            
        else:
            raise ValueError("Either transcript or segments must be provided")
        
        response = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Transcript: \n\n{input_text}"}
            ],
            response_format={"type": "json_object"}, 
            temperature=0.1, 
            logprobs=True 
        )
        
        content = response.choices[0].message.content
        data = json.loads(content)

        #Process confidence scores
        if response.choices[0].logprobs:
            raw_logprobs = response.choices[0].logprobs.content
            logprobs_formatted = logprobs_handler.format_logprobs(raw_logprobs)
            confidence_dict = logprobs_handler.process_logprobs(logprobs_formatted)
            
            #Calculate ovr confidence from all available scores
            valid_scores = [v for v in confidence_dict.values() if isinstance(v, (int, float))]
            overall = round(sum(valid_scores) / len(valid_scores), 4) if valid_scores else 0.0
            
            data["confidence"] = {
                "overall": overall,
                "by_token": confidence_dict  
            }
        else:
            data["confidence"] = None
        
        # data["transcript"] = input_text
        # data["source_type"] = source_type  # "conversation" or "transcript"
        return data
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "transcript": transcript or (format_conversation(segments) if segments else ""),
            "error": str(e)
        }