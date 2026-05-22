from transformers import AutoTokenizer

# 1. Tải bộ đếm token của chính Qwen2.5
model_path = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 2. Dán đoạn text bạn muốn kiểm tra vào đây
text_can_kiem_tra = """
<|im_start|>system
You are a helpful and expert medical assistant. Identify the correct response employing a logical and systematic strategy.<|im_end|>
<|im_start|>user
Encountering a medical inquiry alongside several alternatives, your mission is to ascertain the correct solution through a structured methodology. Begin by providing a concise overview of the question's subject matter, followed by a thorough analysis of each provided option. Ultimately, utilize a stepwise analytical approach to arrive at an accurate answer. Then, indicate your final choice decision. Question: A 45-year-old male presents to his primary care physician complaining of drainage from his left great toe. He has had an ulcer on his left great toe for over eight months. He noticed increasing drainage from the ulcer over the past week. His past medical history is notable for diabetes mellitus on insulin complicated by peripheral neuropathy and retinopathy. His most recent hemoglobin A1c was 9.4%. He has a 25 pack-year smoking history. He has multiple sexual partners and does not use condoms. His temperature is 100.8°F (38.2°C), blood pressure is 150/70 mmHg, pulse is 100/min, and respirations are 18/min. Physical examination reveals a 1 cm ulcer on the plantar aspect of the left great toe surrounded by an edematous and erythematous ring. Exposed bone can be palpated with a probe. There are multiple small cuts and bruises on both feet. A bone biopsy reveals abundant gram-negative rods that do not ferment lactose. The pathogen most likely responsible for this patient’s current condition is also strongly associated with which of the following conditions?
Choices:
A. Otitis externa
B. Gastroenteritis
C. Waterhouse-Friedrichsen syndrome
D. Rheumatic fever<|im_end|>
<|im_start|>assistant
A 45-year-old male presents to his primary care physician with drainage from his left great toe. He has had an ulcer on his left great toe for over eight months, which has been increasing in drainage over the past week. He has a history of diabetes mellitus complicated by peripheral neuropathy and retinopathy, and a 25 pack-year smoking history. His most recent hemoglobin A1c was 9.4%. A bone biopsy reveals abundant gram-negative rods that do not ferment lactose, indicating a Pseudomonas aeruginosa infection. The patient's temperature is 100.8°F (38.2°C), blood pressure is 150/70 mmHg, pulse is 100/min, and respirations are 18/min. Physical examination reveals a 1 cm ulcer on the plantar aspect of the left great toe surrounded by an edematous and erythematous ring, and exposed bone can be palpated with a probe. There are multiple small cuts and bruises on both feet. The task is to determine which of the following conditions is strongly associated with the pathogen most likely responsible for this patient's current condition.
Let's break down the options:
Option A: Otitis externa. Pseudomonas aeruginosa is a known cause of otitis externa, especially in swimmers or individuals with frequent ear canal moisture exposure. This condition is an infection of the outer ear canal and is often associated with the production of a blue-green pigment called pyocyanin, which is characteristic of Pseudomonas infections.
Option B: Gastroenteritis. While Pseudomonas aeruginosa can cause gastroenteritis, it is a less common cause than other bacterial pathogens like Salmonella, Campylobacter, and Escherichia coli. This option is less likely to be the correct answer.
Option C: Waterhouse-Friedrichsen syndrome. This is a condition characterized by widespread purpura fulminans in association with meningococcemia or meningitis. It is caused by Neisseria meningitidis and not Pseudomonas aeruginosa. Therefore, this option is not the correct answer.
Option D: Rheumatic fever. Rheumatic fever is an autoimmune reaction to a Streptococcal infection, not a Pseudomonas infection. Thus, this option is not the correct answer.
Given the clinical presentation, the presence of a gram-negative rod that does not ferment lactose, and the known associations of Pseudomonas infections, the most likely condition associated with the pathogen responsible for this patient's current condition is otitis externa (Option A). Therefore, the correct answer is  A. Otitis externa.
Answer: A.<|im_end|>
"""

# 3. Đếm token
tokens = tokenizer.encode(text_can_kiem_tra)
so_luong_token = len(tokens)

print(f"Đoạn text này dài: {so_luong_token} tokens (Theo chuẩn Qwen2.5)")

if so_luong_token > 2048:
    print("⚠️ CẢNH BÁO: Vượt quá giới hạn 2048 tokens!")
else:
    print("✅ AN TOÀN: Phù hợp để đưa vào huấn luyện.")