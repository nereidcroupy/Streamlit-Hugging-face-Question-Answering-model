from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import streamlit as st 
import string

st.title("Вопросно-ответная система")

st.write('*Заполните все поля ниже*')

question = st.text_input(label="Введите вопрос: ", key='text1')
context = st.text_input(label="Введите контекст вопроса: ", key='text2')

model_name = "AlexKay/xlm-roberta-large-qa-multilingual-finedtuned-ru"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=False)

nlp = pipeline("question-answering", model=model, tokenizer=tokenizer)

def get_answer():
    if question and context:
        input_text = {
            "question": question,
            "context": context
        }
        main_container.write(f"Ответ: {nlp(input_text)['answer'].translate(str.maketrans('','', string.punctuation))}")
        main_container.write(f"Оценка: {nlp(input_text)['score']:.5f}")
    else:
        main_container.write("Пожалуйста, заполните оба поля.")

      
def clear_text():
    st.session_state.text1 = ""
    st.session_state.text2 = ""
   
    
st.button("Получить ответ", on_click=get_answer)

st.write("**Вывод**")
main_container = st.container()
st.button("Очистить", on_click=clear_text)





