import json
from typing import Dict, Any
from openai import OpenAI
import configs

# Groq — OpenAI-compatible API, fast LPU inference
client = OpenAI(
    api_key=configs.GROQ_KEY,
    base_url="https://api.groq.com/openai/v1"
)

MODEL = "llama-3.3-70b-versatile"

def format_conversation(segments: list[dict]) -> str:
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
    
    if current_speaker and current_text:
        conversation.append(f"{current_speaker}: {' '.join(current_text)}")
    
    return "\n\n".join(conversation)


def extract_medical_data(transcript: str = None, segments: list[dict] = None) -> Dict[str, Any]:    
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
            input_text = format_conversation(segments)
        elif transcript:
            input_text = transcript
        else:
            raise ValueError("Either transcript or segments must be provided")
        
        response = client.chat.completions.create(
            model=MODEL, 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Transcript: \n\n{input_text}"}
            ],
            temperature=0.1, 
        )
        
        content = response.choices[0].message.content.strip()
        
        # Strip markdown code blocks if LLM wraps the JSON
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            content = content.rsplit("```", 1)[0].strip()
        
        data = json.loads(content)
        return data
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "transcript": transcript or (format_conversation(segments) if segments else ""),
            "error": str(e)
        }