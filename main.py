import main as st
import chromadb
import os
import google.generativeai as genai
from deep_translator import GoogleTranslator
from dotenv import load_dotenv


load_dotenv()


gemini_api_key = os.getenv("GEMINI_API_KEY")


CHROMA_PATH = "Hesaplama Chroma Database"


chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name="hesaplama_data")


genai.configure(api_key=gemini_api_key)
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}


system_prompt = """
Siz, Hesaplama Teorisi konusunda uzmanlaşmış bir yapay zeka destekli eğitim asistanısınız.

Temel Rol ve Amaçlarınız:
1. Hesaplama modelleri (ör. Turing makineleri, deterministik ve nondeterministik modeller), dillerin sınıflandırılması (ör. düzenli diller, bağlamdan bağımsız diller), karmaşıklık teorisi ve algoritmaların temel ilkeleri hakkında net, doğru ve özlü açıklamalar sağlamak.
2. Öğrenicilerin seviyesine uygun şekilde içeriği uyarlayarak başlangıç seviyesinden ileri seviyeye kadar her düzeydeki öğreniciye hizmet etmek.

Yanıtlama Stratejiniz:
- Sağlanan bağlam ve dokümanları dikkate alarak en alakalı ve doğru bilgiyi sağlayın.
- Teknik terimleri açık ve anlaşılır bir şekilde açıklayın.
- Karmaşık kavramları somut örnekler ve görselleştirmelerle destekleyerek kolay anlaşılır hale getirin.
- Farklı düzeydeki öğreniciler için uygun anlatım tarzı benimseyin (örneğin, yeni başlayanlar için sade bir dil, ileri düzey öğrenciler için detaylı ve teknik açıklamalar).

Yanıt Verme İlkeleri:
- Dostane, profesyonel ve teşvik edici bir üslup kullanın.
- Bilimsel doğruluğu ve tutarlılığı her zaman ön planda tutun.
- Sorulara açık, mantıklı ve yapılandırılmış yanıtlar verin.

Önemli Not: 
- Sağlanan bağlamda belirli bir bilgi yoksa, bu durumu belirtin ve genel bilginize başvurarak açıklamalar yapın.
"""


model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction=system_prompt
)






st.set_page_config(page_title="Hesaplama Teorisi Eğitim Asistanı")
st.title("Hesaplama Teorisi Eğitim Asistanı")
st.write("Bu asistan, hesaplama teorisi ilgili sorularınıza yanıt vermek için tasarlanmıştır. Sorularınızı aşağıdaki kutuya yazabilirsiniz.")


if "messages" not in st.session_state:
    st.session_state["messages"] = []


for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])


if user_query := st.chat_input("Soru yazın ve Enter'a basın..."):
    
    with st.chat_message("user"):
        st.write(user_query)

    
    st.session_state["messages"].append({"role": "user", "content": user_query})

    try:
        
        user_query_en = GoogleTranslator(source='auto', target='en').translate(user_query)

        
        results = collection.query(query_texts=[user_query_en], n_results=3)

       
        document_data = results['documents'][0] if results["documents"] else ""
        contextual_prompt = f"{system_prompt}\n--------------------\n Soru: {user_query} \n--------------------\n The Context:\n{document_data}"

       
        chat_session = model.start_chat(history=[])
        chat_session.system_instruction = contextual_prompt
        response = chat_session.send_message(user_query)

       
        with st.chat_message("assistant"):
            if document_data:
                st.write("**Bağlam (Context):**")
                st.write(document_data)
            st.write("**Yanıt:**")
            st.write(response.text)

        
        st.session_state["messages"].append({"role": "assistant", "content": response.text})

    except Exception as e:
        st.error(f"Bir hata oluştu: {e}")
