from transformers import AutoTokenizer

# 1. Tải bộ đếm token của chính Qwen2.5
model_path = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 2. Dán đoạn text bạn muốn kiểm tra vào đây
text_can_kiem_tra = """
The Rural Trauma Team Development Course (RTTDC) is designed to teach knowledge and skills for the initial assessment and stabilization of trauma patients in resource-limited environments. In this study, patients from six rural hospitals that participated in an RTTDC course were compared with a control group of similar centers that did not participate in the course. The results showed that the RTTDC group experienced an overall 61-minute reduction in referring hospital emergency department (ED) length of stay (LOS) compared with the control group. The RTTDC group also showed a 41-minute reduction in time to call for transfer compared with controls. There were no differences in the secondary outcomes of pretransfer computed tomography (CT) scanning rates or mortality.
The study suggests that RTTDC training can decrease time to transfer for trauma patients, as shown by the reduction in referring hospital ED LOS and time to call for transfer in the RTTDC group compared with the control group. However, it is important to note that further research and educational efforts should focus on decreasing unnecessary imaging prior to transfer and improving mortality rates.
Answer: Yes.
"""

# 3. Đếm token
tokens = tokenizer.encode(text_can_kiem_tra)
so_luong_token = len(tokens)

print(f"Đoạn text này dài: {so_luong_token} tokens (Theo chuẩn Qwen2.5)")

if so_luong_token > 2048:
    print("⚠️ CẢNH BÁO: Vượt quá giới hạn 2048 tokens!")
else:
    print("✅ AN TOÀN: Phù hợp để đưa vào huấn luyện.")