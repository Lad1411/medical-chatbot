from transformers import AutoTokenizer

# 1. Tải bộ đếm token của chính Qwen2.5
model_path = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 2. Dán đoạn text bạn muốn kiểm tra vào đây
text_can_kiem_tra = """
The question presents a case of a 3-week-old infant who is being seen by the pediatrician due to concerns about his feeding habits. The infant was born without complications and has not had any medical problems up until this time. However, for the past 4 days, the infant has been fussy, regurgitating all of his feeds, and the vomit is yellow in color. On physical examination, the child's abdomen is minimally distended, but no other abnormalities are found. The infant's symptoms, including fussiness, regurgitation of all feeds, and yellow vomit, suggest a problem with the digestive system. The fact that the vomit is yellow indicates that it contains bile, which is produced by the liver and stored in the gallbladder. Bile is normally released into the small intestine to aid in digestion. However, in this case, it is being regurgitated, which suggests that there may be an obstruction in the upper gastrointestinal tract.
The possible embryologic errors that could account for this presentation are:
A. Abnormal migration of ventral pancreatic bud: During embryonic development, the pancreas forms from two buds that arise from the foregut. The ventral bud migrates to fuse with the dorsal bud, forming the pancreas. If the ventral bud does not migrate properly, it can result in pancreatic tissue being located outside of the pancreas, which can cause obstruction of the gastrointestinal tract.
B. Complete failure of proximal duodenum to recanalize: During embryonic development, the lumen of the gastrointestinal tract is initially solid, but it later becomes hollow through a process called recanalization. If the proximal duodenum fails to recanalize completely, it can result in an obstruction of the gastrointestinal tract.
C. Abnormal hypertrophy of the pylorus: The pylorus is the muscular valve that connects the stomach to the small intestine. If the pylorus becomes abnormally thickened, it can cause an obstruction of the gastrointestinal tract.
D. Failure of lateral body folds to move ventrally and fuse in the midline: During embryonic development, the lateral body folds move ventrally and fuse in the midline to form the anterior abdominal wall. If this process fails, it can result in an omphalocele, a congenital defect in which the intestines or other abdominal organs protrude through the navel.
Based on the information provided, the most likely embryologic error that could account for this presentation is abnormal hypertrophy of the pylorus (option C). This is because the infant's symptoms suggest an obstruction in the upper gastrointestinal tract, and abnormal thickening of the pylorus can cause such an obstruction.
Answer: C.
"""

# 3. Đếm token
tokens = tokenizer.encode(text_can_kiem_tra)
so_luong_token = len(tokens)

print(f"Đoạn text này dài: {so_luong_token} tokens (Theo chuẩn Qwen2.5)")

if so_luong_token > 2048:
    print("⚠️ CẢNH BÁO: Vượt quá giới hạn 2048 tokens!")
else:
    print("✅ AN TOÀN: Phù hợp để đưa vào huấn luyện.")